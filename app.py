import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace
from supabase import create_client

# -------------------------
# SUPABASE CONFIG
# -------------------------

SUPABASE_URL = os.environ.get("sdmonyqbrckcblohqzvw")
SUPABASE_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNkbW9ueXFicmNrY2Jsb2hxenZ3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMwNjI2MzUsImV4cCI6MjA4ODYzODYzNX0.vRTzQS8ISS-KxJMNZ4dfCZyiml8d7u0D16OuZJRgwfk")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("🔐 Face Recognition System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Register Face", "Detect Face"]
)

# -------------------------
# REGISTER FACE
# -------------------------

if menu == "Register Face":

    st.header("Register Face")

    name = st.text_input("Enter Name")

    img_file = st.camera_input("Capture Image")

    if img_file and name:

        with open("register.jpg", "wb") as f:
            f.write(img_file.getbuffer())

        # upload image to supabase storage
        with open("register.jpg", "rb") as f:
            supabase.storage.from_("faces").upload(
                f"{name}.jpg",
                f
            )

        st.success("Face Registered Successfully")

# -------------------------
# DETECT FACE
# -------------------------

if menu == "Detect Face":

    st.header("Detect Face")

    img_file = st.camera_input("Capture Image")

    if img_file:

        with open("test.jpg", "wb") as f:
            f.write(img_file.getbuffer())

        # example comparison (single user)
        result = DeepFace.verify(
            img1_path="test.jpg",
            img2_path="register.jpg",
            model_name="Facenet"
        )

        if result["verified"]:
            st.success("✅ MATCH FOUND")
        else:
            st.error("❌ NO MATCH FOUND")
SUPABASE_URL = st.secrets["sdmonyqbrckcblohqzvw"]
SUPABASE_KEY = st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNkbW9ueXFicmNrY2Jsb2hxenZ3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMwNjI2MzUsImV4cCI6MjA4ODYzODYzNX0.vRTzQS8ISS-KxJMNZ4dfCZyiml8d7u0D16OuZJRgwfk"]

