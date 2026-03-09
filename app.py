import streamlit as st
import cv2
import numpy as np
from supabase import create_client

# -----------------------------
# SUPABASE CONNECTION
# -----------------------------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("🔐 Face Recognition System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Register Face", "Detect Face"]
)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def extract_feature(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (100,100))

    feature = resized.flatten()

    return feature.tolist()

# -----------------------------
# REGISTER FACE
# -----------------------------

if menu == "Register Face":

    st.header("Register New Face")

    name = st.text_input("Enter Name")

    img_file = st.camera_input("Capture Face")

    if img_file and name:

        bytes_data = img_file.getvalue()

        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)

        image = cv2.imdecode(file_bytes, 1)

        feature = extract_feature(image)

        supabase.table("faces").insert({

            "name": name,
            "feature": feature

        }).execute()

        st.success("Face Registered Successfully")

# -----------------------------
# DETECT FACE
# -----------------------------

if menu == "Detect Face":

    st.header("Face Detection")

    img_file = st.camera_input("Capture Image")

    if img_file:

        bytes_data = img_file.getvalue()

        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)

        image = cv2.imdecode(file_bytes, 1)

        new_feature = np.array(extract_feature(image))

        data = supabase.table("faces").select("*").execute()

        found = False

        for row in data.data:

            stored = np.array(row["feature"])

            distance = np.linalg.norm(new_feature - stored)

            if distance < 2000:

                st.success(f"✅ MATCH FOUND: {row['name']}")

                found = True
                break

        if not found:

            st.error("❌ NO MATCH FOUND")







