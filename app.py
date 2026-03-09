import streamlit as st
import face_recognition
import numpy as np
import cv2
from supabase import create_client

# ------------------------
# SUPABASE CONNECTION
# ------------------------

SUPABASE_URL = "PASTE_URL_HERE"
SUPABASE_KEY = "PASTE_KEY_HERE"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.title("Face Recognition System")

menu = st.sidebar.selectbox(
    "Select",
    ["Register Face", "Detect Face"]
)

# ------------------------
# FACE EMBEDDING FUNCTION
# ------------------------

def get_embedding(image):

    faces = face_recognition.face_locations(image)

    if len(faces) == 0:
        return None

    encoding = face_recognition.face_encodings(
        image,
        faces
    )[0]

    return encoding.tolist()

# ------------------------
# REGISTER FACE
# ------------------------

if menu == "Register Face":

    name = st.text_input("Enter Name")

    img_file = st.camera_input("Capture Face")

    if img_file:

        bytes_data = img_file.getvalue()

        file_bytes = np.asarray(
            bytearray(bytes_data),
            dtype=np.uint8
        )

        img = cv2.imdecode(file_bytes, 1)

        embedding = get_embedding(img)

        if embedding is None:

            st.error("No face detected")

        else:

            supabase.table("faces").insert({

                "name": name,
                "embedding": embedding

            }).execute()

            st.success("Face Registered")

# ------------------------
# DETECT FACE
# ------------------------

if menu == "Detect Face":

    img_file = st.camera_input("Capture Image")

    if img_file:

        bytes_data = img_file.getvalue()

        file_bytes = np.asarray(
            bytearray(bytes_data),
            dtype=np.uint8
        )

        img = cv2.imdecode(file_bytes, 1)

        new_embedding = get_embedding(img)

        if new_embedding is None:

            st.error("No face detected")

        else:

            data = supabase.table("faces").select("*").execute()

            found = False

            for row in data.data:

                stored = np.array(row["embedding"])

                match = face_recognition.compare_faces(
                    [stored],
                    new_embedding
                )

                if match[0]:

                    st.success(
                        f"Match Found: {row['name']}"
                    )

                    found = True
                    break

            if not found:

                st.error("No Match")