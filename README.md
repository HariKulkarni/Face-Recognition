# 📸 Face Recognition Attendance System

This project is a **Streamlit-based facial recognition attendance system** that enables automated student attendance tracking for multiple subjects using computer vision and Excel reporting.

---

## 🚀 Features

- **Face Data Collection**: Register students by capturing face samples using a webcam.
- **Model Training**: Train an LBPH (Local Binary Patterns Histograms) face recognition model.
- **Automated Attendance**: Recognize faces and mark attendance per subject and date.
- **Excel Integration**: Automatically updates structured Excel sheets with attendance, totals, and percentages.
- **Monthly Reports**: Generate and view monthly attendance summaries and subject-wise analytics.
- **User Interface**: Fully interactive via a web UI using Streamlit.

---

## 🛠️ Technologies Used

- Python
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- Streamlit
- NumPy, Pandas
- OpenPyXL
- Excel (as backend storage)

---

## 🗂️ Project Structure

```
├── face_data/                # Directory to store collected face images
├── attendance_excel/         # Directory for generated monthly Excel reports
├── id_name.txt               # Mapping of student IDs to names
├── trainer.yml               # Trained face recognition model
├── gui2.py                   # Main Streamlit application
└── README.md                 # Project documentation
```

---

## ⚙️ How to Run

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>requirements.txt (suggested)</strong></summary>

```text
opencv-python
opencv-contrib-python
streamlit
pandas
numpy
openpyxl
```
</details>

2. **Run the Streamlit App**

```bash
streamlit run gui2.py
```

---

## 📋 Workflow

1. **👤 Collect Face Data**
   - Enter student ID and name
   - Capture 30 face samples using your webcam

2. **🤖 Train Model**
   - Use collected data to train the LBPH face recognizer

3. **✅ Mark Attendance**
   - Select subject and date
   - Start camera and let the system recognize student faces
   - Attendance is automatically logged in the Excel sheet

4. **📊 View Results**
   - View summary reports, subject-wise statistics, or download detailed records

---

## 📁 Output

- Attendance is saved to:
  ```
  attendance_excel/<Month>_<Year>_attendance.xlsx
  ```

- Includes:
  - Daily attendance per subject
  - Subject-wise totals
  - Overall attendance percentage
  - Monthly summary statistics

---

## ✅ Status Indicators

- Face data and model existence
- Excel file readiness
- Registered students count

---

## 📌 Notes

- Make sure your webcam is working.
- Ensure `opencv-contrib-python` is installed for `cv2.face` module.
- Close the Excel file while marking attendance to avoid saving errors.

---

## 📧 Author

- Developed by **Harish Kulkarni**  
- For educational and institutional attendance automation.