import streamlit as st
import cv2
import numpy as np
import os
from supabase import create_client

# -------------------
# SUPABASE
# -------------------

SUPABASE_URL = os.environ.get("sdmonyqbrckcblohqzvw")
SUPABASE_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNkbW9ueXFicmNrY2Jsb2hxenZ3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMwNjI2MzUsImV4cCI6MjA4ODYzODYzNX0.vRTzQS8ISS-KxJMNZ4dfCZyiml8d7u0D16OuZJRgwfk")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------
# PAGE
# -------------------

st.title("Simple Face Recognition")

menu = st.sidebar.selectbox(
    "Menu",
    ["Register", "Detect"]
)

# -------------------
# FACE FEATURE
# -------------------

def extract_feature(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (100,100))

    feature = resized.flatten()

    return feature.tolist()

# -------------------
# REGISTER
# -------------------

if menu == "Register":

    name = st.text_input("Enter Name")

    img = st.camera_input("Capture Face")

    if img and name:

        bytes_data = img.getvalue()

        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)

        frame = cv2.imdecode(file_bytes, 1)

        feature = extract_feature(frame)

        supabase.table("faces").insert({
            "name": name,
            "feature": feature
        }).execute()

        st.success("Face Registered")

# -------------------
# DETECT
# -------------------

if menu == "Detect":

    img = st.camera_input("Capture Face")

    if img:

        bytes_data = img.getvalue()

        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)

        frame = cv2.imdecode(file_bytes, 1)

        new_feature = np.array(extract_feature(frame))

        data = supabase.table("faces").select("*").execute()

        found = False

        for row in data.data:

            stored = np.array(row["feature"])

            distance = np.linalg.norm(new_feature - stored)

            if distance < 2000:

                st.success(f"Match Found: {row['name']}")

                found = True

                break

        if not found:

            st.error("No Match Found")

