
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import os
from datetime import datetime, date
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import plotly.express as px

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RentEase | Rental Property Management",
    page_icon="🏠",
    layout="wide"
)

DB = "rental_app.db"
CSV_FILE = "rental_data.csv"

# =========================================================
# 2. DATABASE CONNECTION
# =========================================================

def get_connection():
    return sqlite3.connect(DB)

def initialize_database():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            phone TEXT,
            password TEXT,
            role TEXT DEFAULT 'user'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS properties(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER,
            title TEXT,
            property_type TEXT,
            location TEXT,
            area_sqft REAL,
            bedrooms INTEGER,
            bathrooms INTEGER,
            furnishing TEXT,
            rent REAL,
            description TEXT,
            status TEXT DEFAULT 'Available',
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS favorites(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            property_id INTEGER,
            UNIQUE(user_id, property_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS bookings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            property_id INTEGER,
            visit_date TEXT,
            message TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS payments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            property_id INTEGER,
            amount REAL,
            payment_date TEXT,
            payment_method TEXT,
            payment_status TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            location TEXT,
            property_type TEXT,
            area_sqft REAL,
            bedrooms INTEGER,
            predicted_rent REAL,
            category TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            message TEXT,
            created_at TEXT
        )
    """)

    # Default administrator account
    cur.execute(
        "SELECT id FROM users WHERE email=?",
        ("admin@rentease.com",)
    )

    if not cur.fetchone():
        password = hashlib.sha256(
            "Admin@12345".encode()
        ).hexdigest()

        cur.execute("""
            INSERT INTO users(name,email,phone,password,role)
            VALUES(?,?,?,?,?)
        """, (
            "RentEase Admin",
            "admin@rentease.com",
            "0000000000",
            password,
            "admin"
        ))

    conn.commit()
    conn.close()


initialize_database()

# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def execute_query(query, params=(), fetch=False):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params)

    result = cur.fetchall() if fetch else None

    conn.commit()
    conn.close()

    return result


def read_data():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)

            required = [
                "area_sqft", "bedrooms", "bathrooms",
                "location", "furnishing",
                "property_type", "rent"
            ]

            if all(col in df.columns for col in required):
                df = df[required].copy()

                df["rent"] = pd.to_numeric(
                    df["rent"], errors="coerce"
                )

                df["area_sqft"] = pd.to_numeric(
                    df["area_sqft"], errors="coerce"
                )

                df = df.dropna()

                if len(df) >= 5:
                    return df

        except Exception:
            pass

    # Sample dataset if CSV is unavailable
    return pd.DataFrame({
        "area_sqft": [500, 700, 900, 1100, 1300, 1500,
                      1800, 2000, 2200, 2500, 800, 1200],
        "bedrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 3],
        "bathrooms": [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 2],
        "location": [
            "City Center", "Suburb", "Near College",
            "IT Park", "City Center", "Suburb",
            "Near College", "IT Park", "City Center",
            "Suburb", "IT Park", "Near College"
        ],
        "furnishing": [
            "Furnished", "Unfurnished", "Semi-Furnished",
            "Furnished", "Semi-Furnished", "Unfurnished",
            "Furnished", "Semi-Furnished", "Furnished",
            "Unfurnished", "Semi-Furnished", "Furnished"
        ],
        "property_type": [
            "Apartment", "Studio", "Apartment",
            "Independent House", "Apartment", "Villa",
            "Independent House", "Villa", "Apartment",
            "Independent House", "Studio", "Villa"
        ],
        "rent": [
            12000, 15000, 18000, 22000, 25000, 28000,
            35000, 40000, 45000, 50000, 14000, 30000
        ]
    })


df = read_data()

# =========================================================
# 4. MACHINE LEARNING MODELS
# =========================================================

features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "location",
    "furnishing",
    "property_type"
]

numeric_features = [
    "area_sqft", "bedrooms", "bathrooms"
]

categorical_features = [
    "location", "furnishing", "property_type"
]

preprocessor = ColumnTransformer([
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    ("numeric", "passthrough", numeric_features)
])

rent_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

category_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

df["rent_category"] = pd.cut(
    df["rent"],
    bins=[-np.inf, 15000, 30000, np.inf],
    labels=["Budget", "Moderate", "Premium"]
).astype(str)

rent_model.fit(df[features], df["rent"])

category_model.fit(
    df[features],
    df["rent_category"]
)

# =========================================================
# 5. SESSION STATE
# =========================================================

if "user" not in st.session_state:
    st.session_state.user = None

if "role" not in st.session_state:
    st.session_state.role = None

if "page" not in st.session_state:
    st.session_state.page = "Home"


# =========================================================
# 6. LOGIN AND REGISTRATION
# =========================================================

def register_user(name, email, phone, password):

    hashed = hashlib.sha256(
        password.encode()
    ).hexdigest()

    try:
        execute_query("""
            INSERT INTO users(name,email,phone,password,role)
            VALUES(?,?,?,?,?)
        """, (name, email, phone, hashed, "user"))

        return True

    except sqlite3.IntegrityError:
        return False


def login_user(email, password):

    hashed = hashlib.sha256(
        password.encode()
    ).hexdigest()

    result = execute_query("""
        SELECT id,name,role
        FROM users
        WHERE email=? AND password=?
    """, (email, hashed), fetch=True)

    if result:
        st.session_state.user = {
            "id": result[0][0],
            "name": result[0][1],
            "email": email
        }

        st.session_state.role = result[0][2]
        return True

    return False


# =========================================================
# 7. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🏠 RentEase")

st.sidebar.caption(
    "Smart Rental Property Management System"
)

if st.session_state.user:
    st.sidebar.success(
        f"Welcome, {st.session_state.user['name']}"
    )

pages = [
    "Home",
    "Property Listings",
    "Rental Price Prediction",
    "Add Property",
    "My Dashboard",
    "Bookings",
    "Payments",
    "Analytics & Reports",
    "About Us",
    "Contact Us",
    "Login / Create Account"
]

if st.session_state.role == "admin":
    pages.append("Admin Panel")

if st.session_state.user is None:
    selected_page = "Login / Create Account"
    st.session_state.page = selected_page
else:
    selected_page = st.sidebar.radio(
        "Select a page:",
        pages,
        index=pages.index(st.session_state.page)
        if st.session_state.page in pages else 0
    )
    st.session_state.page = selected_page

if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.page = "Home"
    st.rerun()


# =========================================================
# 8. HOME PAGE
# =========================================================

if selected_page == "Home":

    st.title("🏠 Welcome to RentEase")

    st.subheader(
        "Find Your Perfect Rental Property"
    )

    st.write(
        "A smart platform for property searching, "
        "rental price prediction and property management."
    )

    col1, col2, col3 = st.columns(3)

    properties = execute_query(
        "SELECT COUNT(*) FROM properties",
        fetch=True
    )[0][0]

    col1.metric("Listed Properties", properties)
    col2.metric("Dataset Records", len(df))
    col3.metric("Rental Categories", 3)

    st.divider()

    st.subheader("What can you do with RentEase?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🔍 Search Properties
        Find properties using location, budget
        and property type.

        ### 🤖 Rental Price Prediction
        Estimate monthly rent using machine learning.

        ### ❤️ Save Favourite Properties
        Keep your preferred rental properties.
        """)

    with col2:
        st.markdown("""
        ### 🏘️ Manage Properties
        Add and manage your rental listings.

        ### 📅 Property Booking
        Request property visits.

        ### 📊 Dashboard
        View rental information and reports.
        """)

    st.divider()

    st.subheader("Featured Rental Properties")

    featured = execute_query("""
        SELECT id,title,location,property_type,
               area_sqft,bedrooms,rent
        FROM properties
        WHERE status='Available'
        ORDER BY id DESC LIMIT 6
    """, fetch=True)

    if featured:
        for item in featured:
            with st.container(border=True):
                st.markdown(f"### 🏠 {item[1]}")
                st.write(f"📍 Location: {item[2]}")
                st.write(f"Property type: {item[3]}")
                st.write(f"Area: {item[4]} sq.ft")
                st.write(f"Bedrooms: {item[5]}")
                st.success(f"Monthly rent: ₹{item[6]:,.0f}")
    else:
        st.info("No properties listed yet. Register and add a property.")

# =========================================================
# 9. LOGIN / CREATE ACCOUNT
# =========================================================

elif selected_page == "Login / Create Account":

    st.title("🔐 Account Access")

    login_tab, register_tab = st.tabs([
        "Login",
        "Create Account"
    ])

    with login_tab:

        st.subheader("Login to RentEase")

        with st.form("login_form"):

            email = st.text_input(
                "Email address",
                placeholder="Enter your registered email address"
            )

            password = st.text_input(
                "Password",
                placeholder="Enter your password",
                type="password"
            )

            submit = st.form_submit_button(
                "Login",
                use_container_width=True
            )

            if submit:

                if email.strip() and password:

                    if login_user(email.strip().lower(), password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Incorrect email or password.")

                else:
                    st.error("Please enter your email and password.")

    with register_tab:

        st.subheader("Create Your Account")

        with st.form("register_form"):

            name = st.text_input(
                "Full Name",
                placeholder="Example: Lohitha Chowdary"
            )

            email = st.text_input(
                "Email Address",
                placeholder="Example: name@gmail.com"
            )

            phone = st.text_input(
                "Mobile Number",
                placeholder="Enter your 10-digit mobile number"
            )

            password = st.text_input(
                "Create Password",
                placeholder="Create a password",
                type="password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                placeholder="Re-enter your password",
                type="password"
            )

            register = st.form_submit_button(
                "Create Account",
                use_container_width=True
            )

            if register:

                if not all([
                    name.strip(),
                    email.strip(),
                    phone.strip(),
                    password,
                    confirm_password
                ]):
                    st.error("Please fill in all required fields.")

                elif "@" not in email or "." not in email:
                    st.error("Please enter a valid email address.")

                elif not phone.isdigit() or len(phone) != 10:
                    st.error("Enter a valid 10-digit mobile number.")

                elif len(password) < 6:
                    st.error("Password must contain at least 6 characters.")

                elif password != confirm_password:
                    st.error("Passwords do not match.")

                else:
                    success = register_user(
                        name.strip(),
                        email.strip().lower(),
                        phone,
                        password
                    )

                    if success:
                        st.success("Account created! Please log in.")
                    else:
                        st.error("This email is already registered.")

# =========================================================
# 10. PROPERTY LISTINGS
# =========================================================

elif selected_page == "Property Listings":

    st.title("🔍 Search Rental Properties")

    st.write(
        "Enter your preferences to find a suitable rental property."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        location_filter = st.selectbox(
            "Select Location",
            ["All"] + sorted(df["location"].unique().tolist())
        )

    with col2:
        type_filter = st.selectbox(
            "Select Property Type",
            ["All"] + sorted(df["property_type"].unique().tolist())
        )

    with col3:
        max_budget = st.number_input(
            "Maximum Monthly Budget (₹)",
            min_value=0,
            value=100000,
            step=5000
        )

    query = """
        SELECT id,title,location,property_type,
               area_sqft,bedrooms,bathrooms,
               furnishing,rent,description,owner_id
        FROM properties
        WHERE status='Available' AND rent<=?
    """

    params = [max_budget]

    if location_filter != "All":
        query += " AND location=?"
        params.append(location_filter)

    if type_filter != "All":
        query += " AND property_type=?"
        params.append(type_filter)

    results = execute_query(query, tuple(params), fetch=True)

    st.subheader(f"Available Properties: {len(results)}")

    if results:

        for p in results:

            with st.container(border=True):

                st.subheader(f"🏠 {p[1]}")

                c1, c2 = st.columns(2)

                with c1:
                    st.write(f"📍 Location: {p[2]}")
                    st.write(f"Property Type: {p[3]}")
                    st.write(f"Area: {p[4]} sq.ft")
                    st.write(f"Bedrooms: {p[5]}")

                with c2:
                    st.write(f"Bathrooms: {p[6]}")
                    st.write(f"Furnishing: {p[7]}")
                    st.write(f"Description: {p[9]}")

                st.success(f"Monthly Rent: ₹{p[8]:,.0f}")

                if st.session_state.user:

                    if st.button(
                        "❤️ Add to Favourites",
                        key=f"fav_{p[0]}"
                    ):

                        try:
                            execute_query("""
                                INSERT INTO favorites(user_id,property_id)
                                VALUES(?,?)
                            """, (
                                st.session_state.user["id"],
                                p[0]
                            ))

                            st.success("Added to favourites!")

                        except sqlite3.IntegrityError:
                            st.info("Already in your favourites.")

# =========================================================
# 11. RENTAL PRICE PREDICTION
# =========================================================

elif selected_page == "Rental Price Prediction":

    st.title("🤖 AI Rental Price Prediction")

    st.write(
        "Enter the property details below to estimate "
        "the expected monthly rental price."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            area = st.number_input(
                "Property Area (square feet)",
                min_value=100,
                max_value=20000,
                value=1000,
                help="Enter the total built-up area in square feet."
            )

            bedrooms = st.selectbox(
                "Number of Bedrooms",
                [1, 2, 3, 4, 5, 6]
            )

            bathrooms = st.selectbox(
                "Number of Bathrooms",
                [1, 2, 3, 4, 5, 6]
            )

            location = st.selectbox(
                "Property Location",
                sorted(df["location"].unique().tolist())
            )

        with col2:

            furnishing = st.selectbox(
                "Furnishing Status",
                ["Furnished", "Semi-Furnished", "Unfurnished"]
            )

            property_type = st.selectbox(
                "Property Type",
                sorted(df["property_type"].unique().tolist())
            )

        predict = st.form_submit_button(
            "Predict Monthly Rent",
            use_container_width=True
        )

    if predict:

        input_data = pd.DataFrame([{
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": location,
            "furnishing": furnishing,
            "property_type": property_type
        }])

        predicted_rent = max(
            0,
            float(rent_model.predict(input_data)[0])
        )

        predicted_category = category_model.predict(
            input_data
        )[0]

        st.divider()

        st.subheader("Prediction Results")

        c1, c2 = st.columns(2)

        c1.metric(
            "Estimated Monthly Rent",
            f"₹{predicted_rent:,.0f}"
        )

        c2.metric(
            "Rental Category",
            predicted_category
        )

        st.success(
            f"Expected monthly rent: ₹{predicted_rent:,.0f}"
        )

        st.info(
            "This estimate is generated by a Random Forest "
            "machine learning model using the available dataset. "
            "Actual rent may differ."
        )

        if st.session_state.user:

            execute_query("""
                INSERT INTO predictions(
                    user_id,location,property_type,
                    area_sqft,bedrooms,predicted_rent,
                    category,created_at
                )
                VALUES(?,?,?,?,?,?,?,?)
            """, (
                st.session_state.user["id"],
                location,
                property_type,
                area,
                bedrooms,
                predicted_rent,
                predicted_category,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ))

# =========================================================
# 12. ADD PROPERTY
# =========================================================

elif selected_page == "Add Property":

    st.title("🏘️ Add Your Rental Property")

    if not st.session_state.user:

        st.warning("Please log in or create an account to add a property.")

    else:

        st.write(
            "Fill in the property information to publish your rental listing."
        )

        with st.form("add_property_form"):

            title = st.text_input(
                "Property Title",
                placeholder="Example: 2BHK Apartment in City Center"
            )

            col1, col2 = st.columns(2)

            with col1:

                property_type = st.selectbox(
                    "Property Type",
                    ["Apartment", "Independent House", "Villa", "Studio"]
                )

                location = st.text_input(
                    "Property Location",
                    placeholder="Example: Chennai, Tambaram"
                )

                area = st.number_input(
                    "Built-up Area (sq.ft)",
                    min_value=100,
                    value=1000
                )

                bedrooms = st.number_input(
                    "Number of Bedrooms",
                    min_value=1,
                    max_value=20,
                    value=2
                )

            with col2:

                bathrooms = st.number_input(
                    "Number of Bathrooms",
                    min_value=1,
                    max_value=20,
                    value=2
                )

                furnishing = st.selectbox(
                    "Furnishing Status",
                    ["Furnished", "Semi-Furnished", "Unfurnished"]
                )

                rent = st.number_input(
                    "Expected Monthly Rent (₹)",
                    min_value=1000,
                    value=15000,
                    step=1000
                )

            description = st.text_area(
                "Property Description",
                placeholder="Describe the property, facilities, nearby transport, parking, etc."
            )

            submit = st.form_submit_button(
                "Publish Property",
                use_container_width=True
            )

            if submit:

                if not title.strip() or not location.strip():
                    st.error("Please enter the property title and location.")

                else:

                    execute_query("""
                        INSERT INTO properties(
                            owner_id,title,property_type,
                            location,area_sqft,bedrooms,
                            bathrooms,furnishing,rent,
                            description,status,created_at
                        )
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        st.session_state.user["id"],
                        title,
                        property_type,
                        location,
                        area,
                        bedrooms,
                        bathrooms,
                        furnishing,
                        rent,
                        description,
                        "Available",
                        datetime.now().strftime("%Y-%m-%d")
                    ))

                    st.success("Your property has been published!")

# =========================================================
# 13. MY DASHBOARD
# =========================================================

elif selected_page == "My Dashboard":

    st.title("📊 My Dashboard")

    if not st.session_state.user:

        st.warning("Please log in to view your dashboard.")

    else:

        uid = st.session_state.user["id"]

        my_properties = execute_query("""
            SELECT COUNT(*) FROM properties WHERE owner_id=?
        """, (uid,), fetch=True)[0][0]

        my_favorites = execute_query("""
            SELECT COUNT(*) FROM favorites WHERE user_id=?
        """, (uid,), fetch=True)[0][0]

        my_bookings = execute_query("""
            SELECT COUNT(*) FROM bookings WHERE user_id=?
        """, (uid,), fetch=True)[0][0]

        c1, c2, c3 = st.columns(3)

        c1.metric("My Properties", my_properties)
        c2.metric("Favourite Properties", my_favorites)
        c3.metric("My Bookings", my_bookings)

        st.divider()

        st.subheader("My Listed Properties")

        listings = execute_query("""
            SELECT id,title,location,rent,status
            FROM properties WHERE owner_id=?
        """, (uid,), fetch=True)

        if listings:
            st.dataframe(pd.DataFrame(
                listings,
                columns=[
                    "Property ID", "Title",
                    "Location", "Monthly Rent", "Status"
                ]
            ), use_container_width=True)
        else:
            st.info("You have not listed any properties.")

        st.subheader("My Favourite Properties")

        favs = execute_query("""
            SELECT p.title,p.location,p.rent
            FROM favorites f
            JOIN properties p ON f.property_id=p.id
            WHERE f.user_id=?
        """, (uid,), fetch=True)

        if favs:
            st.dataframe(pd.DataFrame(
                favs,
                columns=["Property", "Location", "Monthly Rent"]
            ), use_container_width=True)
        else:
            st.info("You have not saved any favourite properties.")

# =========================================================
# 14. BOOKINGS
# =========================================================

elif selected_page == "Bookings":

    st.title("📅 Property Visit Bookings")

    if not st.session_state.user:

        st.warning("Please log in to manage bookings.")

    else:

        uid = st.session_state.user["id"]

        st.subheader("Request a Property Visit")

        available = execute_query("""
            SELECT id,title,location
            FROM properties
            WHERE status='Available'
        """, fetch=True)

        if available:

            property_options = {
                f"{p[1]} - {p[2]} (ID: {p[0]})": p[0]
                for p in available
            }

            with st.form("booking_form"):

                selected = st.selectbox(
                    "Select Property",
                    list(property_options.keys())
                )

                visit_date = st.date_input(
                    "Preferred Visit Date",
                    min_value=date.today()
                )

                message = st.text_area(
                    "Message to Property Owner",
                    placeholder="Example: I would like to visit this property in the evening."
                )

                submit = st.form_submit_button(
                    "Request Visit"
                )

                if submit:

                    execute_query("""
                        INSERT INTO bookings(
                            user_id,property_id,visit_date,
                            message,status,created_at
                        )
                        VALUES(?,?,?,?,?,?)
                    """, (
                        uid,
                        property_options[selected],
                        str(visit_date),
                        message,
                        "Pending",
                        datetime.now().strftime("%Y-%m-%d")
                    ))

                    st.success("Your visit request has been submitted!")

        else:
            st.info("No available properties to book.")

        st.divider()

        st.subheader("My Booking Requests")

        bookings = execute_query("""
            SELECT b.id,p.title,b.visit_date,
                   b.message,b.status
            FROM bookings b
            JOIN properties p ON b.property_id=p.id
            WHERE b.user_id=?
        """, (uid,), fetch=True)

        if bookings:

            st.dataframe(pd.DataFrame(
                bookings,
                columns=[
                    "Booking ID", "Property",
                    "Visit Date", "Message", "Status"
                ]
            ), use_container_width=True)

        else:
            st.info("You have no booking requests.")

# =========================================================
# 15. PAYMENT RECORDS
# =========================================================

elif selected_page == "Payments":

    st.title("💳 Rental Payment Records")

    if not st.session_state.user:

        st.warning("Please log in to access payment records.")

    else:

        uid = st.session_state.user["id"]

        st.info(
            "This section records payment information. "
            "It does not process real online payments."
        )

        st.subheader("Add a Payment Record")

        my_properties = execute_query("""
            SELECT id,title FROM properties
            WHERE owner_id=?
        """, (uid,), fetch=True)

        if my_properties:

            options = {
                f"{p[1]} (ID: {p[0]})": p[0]
                for p in my_properties
            }

            with st.form("payment_form"):

                property_selected = st.selectbox(
                    "Select Your Property",
                    list(options.keys())
                )

                amount = st.number_input(
                    "Payment Amount (₹)",
                    min_value=1,
                    value=10000
                )

                payment_date = st.date_input(
                    "Payment Date",
                    value=date.today()
                )

                method = st.selectbox(
                    "Payment Method",
                    ["Cash", "UPI", "Bank Transfer", "Other"]
                )

                status = st.selectbox(
                    "Payment Status",
                    ["Paid", "Pending"]
                )

                submit = st.form_submit_button(
                    "Save Payment Record"
                )

                if submit:

                    execute_query("""
                        INSERT INTO payments(
                            user_id,property_id,amount,
                            payment_date,payment_method,payment_status
                        )
                        VALUES(?,?,?,?,?,?)
                    """, (
                        uid,
                        options[property_selected],
                        amount,
                        str(payment_date),
                        method,
                        status
                    ))

                    st.success("Payment record saved!")

        else:
            st.info("Add a property first to record its payments.")

        st.divider()

        st.subheader("My Payment History")

        records = execute_query("""
            SELECT p.title,pa.amount,pa.payment_date,
                   pa.payment_method,pa.payment_status
            FROM payments pa
            JOIN properties p ON pa.property_id=p.id
            WHERE pa.user_id=?
        """, (uid,), fetch=True)

        if records:

            st.dataframe(pd.DataFrame(
                records,
                columns=[
                    "Property", "Amount",
                    "Date", "Method", "Status"
                ]
            ), use_container_width=True)

        else:
            st.info("No payment records found.")

# =========================================================
# 16. ANALYTICS AND REPORTS
# =========================================================

elif selected_page == "Analytics & Reports":

    st.title("📈 Rental Analytics & Reports")

    st.subheader("Rental Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", len(df))

    c2.metric(
        "Average Monthly Rent",
        f"₹{df['rent'].mean():,.0f}"
    )

    c3.metric(
        "Average Property Area",
        f"{df['area_sqft'].mean():,.0f} sq.ft"
    )

    st.divider()

    st.subheader("Average Rent by Property Type")

    type_chart = df.groupby(
        "property_type", as_index=False
    )["rent"].mean()

    fig1 = px.bar(
        type_chart,
        x="property_type",
        y="rent",
        title="Average Monthly Rent by Property Type",
        labels={
            "property_type": "Property Type",
            "rent": "Average Rent (₹)"
        }
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Rent Distribution")

    fig2 = px.histogram(
        df,
        x="rent",
        nbins=15,
        title="Monthly Rental Price Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Dataset Preview")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Rental Dataset",
        data=csv,
        file_name="rental_report.csv",
        mime="text/csv"
    )

# =========================================================
# 17. ADMIN PANEL
# =========================================================

elif selected_page == "Admin Panel":

    st.title("⚙️ Administrator Panel")

    if st.session_state.role != "admin":

        st.error("Administrator access only.")

    else:

        st.subheader("Registered Users")

        users = execute_query("""
            SELECT id,name,email,phone,role FROM users
        """, fetch=True)

        if users:
            st.dataframe(pd.DataFrame(
                users,
                columns=["ID", "Name", "Email", "Phone", "Role"]
            ), use_container_width=True)

        st.divider()

        st.subheader("All Rental Properties")

        properties = execute_query("""
            SELECT id,title,location,rent,status
            FROM properties
        """, fetch=True)

        if properties:

            st.dataframe(pd.DataFrame(
                properties,
                columns=[
                    "ID", "Title", "Location",
                    "Rent", "Status"
                ]
            ), use_container_width=True)

            property_id = st.number_input(
                "Enter Property ID to manage",
                min_value=1,
                step=1
            )

            new_status = st.selectbox(
                "Select New Property Status",
                ["Available", "Rented", "Inactive"]
            )

            if st.button("Update Property Status"):

                execute_query("""
                    UPDATE properties SET status=?
                    WHERE id=?
                """, (new_status, property_id))

                st.success("Property status updated!")
                st.rerun()

            if st.button("Delete Selected Property"):

                execute_query(
                    "DELETE FROM properties WHERE id=?",
                    (property_id,)
                )

                st.success("Property deleted.")
                st.rerun()

        st.divider()

        st.subheader("All Booking Requests")

        all_bookings = execute_query("""
            SELECT b.id,u.name,p.title,b.visit_date,
                   b.message,b.status
            FROM bookings b
            JOIN users u ON b.user_id=u.id
            JOIN properties p ON b.property_id=p.id
        """, fetch=True)

        if all_bookings:

            st.dataframe(pd.DataFrame(
                all_bookings,
                columns=[
                    "Booking ID", "Customer",
                    "Property", "Visit Date",
                    "Message", "Status"
                ]
            ), use_container_width=True)

            booking_id = st.number_input(
                "Booking ID to update",
                min_value=1,
                step=1,
                key="admin_booking_id"
            )

            booking_status = st.selectbox(
                "Booking Status",
                ["Pending", "Approved", "Rejected"]
            )

            if st.button("Update Booking"):

                execute_query("""
                    UPDATE bookings SET status=?
                    WHERE id=?
                """, (booking_status, booking_id))

                st.success("Booking updated!")
                st.rerun()

# =========================================================
# 18. ABOUT US
# =========================================================

elif selected_page == "About Us":

    st.title("🏠 About RentEase")

    st.subheader(
        "Smart Rental Property Management System"
    )

    st.write("""
    RentEase is a web-based rental property management
    application designed to simplify property searching,
    rental price estimation and property management.

    The application combines database management,
    machine learning and interactive visualizations
    to provide useful rental information.
    """)

    st.subheader("Main Technologies")

    st.markdown("""
    - Python – Main programming language.
    - Streamlit – Web application interface.
    - Pandas and NumPy – Data processing.
    - Scikit-learn – Machine learning.
    - Random Forest – Rental price prediction and classification.
    - SQLite – User and property database.
    - Plotly – Interactive charts and analytics.
    """)

    st.subheader("Our Main Objectives")

    st.markdown("""
    1. Simplify rental property searching.
    2. Estimate suitable monthly rental prices.
    3. Help owners manage property listings.
    4. Allow tenants to request property visits.
    5. Maintain rental and payment records.
    """)

# =========================================================
# 19. CONTACT US
# =========================================================

elif selected_page == "Contact Us":

    st.title("📞 Contact RentEase")

    st.write(
        "Have a question or suggestion? "
        "Please complete the form below."
    )

    with st.form("contact_form"):

        name = st.text_input(
            "Your Full Name",
            placeholder="Enter your full name"
        )

        email = st.text_input(
            "Your Email Address",
            placeholder="Example: name@gmail.com"
        )

        phone = st.text_input(
            "Your Mobile Number (optional)",
            placeholder="Enter your 10-digit mobile number"
        )

        subject = st.selectbox(
            "Reason for Contact",
            [
                "General Enquiry",
                "Property Listing",
                "Rental Price Prediction",
                "Booking Support",
                "Payment Support",
                "Technical Problem",
                "Other"
            ]
        )

        message = st.text_area(
            "Your Message",
            placeholder="Explain your question or issue here..."
        )

        submit = st.form_submit_button(
            "Submit Feedback",
            use_container_width=True
        )

        if submit:

            if not name.strip() or not email.strip() or not message.strip():

                st.error(
                    "Please fill in your name, email and message."
                )

            elif "@" not in email or "." not in email:

                st.error("Please enter a valid email address.")

            else:

                execute_query("""
                    INSERT INTO feedback(
                        name,email,message,created_at
                    )
                    VALUES(?,?,?,?)
                """, (
                    name,
                    email,
                    f"Subject: {subject}\nPhone: {phone}\n{message}",
                    datetime.now().strftime("%Y-%m-%d %H:%M")
                ))

                st.success(
                    "Thank you! Your feedback has been submitted."
                )

# =========================================================
# 20. FOOTER
# =========================================================

st.divider()

st.caption(
    "© 2026 RentEase | Smart Rental Property Management System"
)

st.caption(
    "Developed using Python, Streamlit, Machine Learning and SQLite."
)
