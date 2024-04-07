# import streamlit as st
# import time
# st.balloons()
# st.progress(10)
# with st.spinner('Wait for it...'):time.sleep(20)

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
rand = np.random.normal(1, 2, size=20)

# Create histogram
fig, ax = plt.subplots()
ax.hist(rand, bins=15)

# Display the plot using Streamlit
st.pyplot(fig)
