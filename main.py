import streamlit as st
import sys

# Debug: show what's in sys.path and if cv2 exists anywhere
import subprocess
result = subprocess.run(["pip", "show", "opencv-python-headless"], capture_output=True, text=True)
st.code(result.stdout or result.stderr)

result2 = subprocess.run(["find", "/", "-name", "cv2*", "-type", "f", "2>/dev/null"], 
                          capture_output=True, text=True, timeout=10)
st.code(result2.stdout[:3000])
st.code("\n".join(sys.path))
