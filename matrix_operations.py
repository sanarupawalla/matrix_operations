import streamlit as st
import numpy as np
import pandas as pd


def parse_matrix(input_str):
    """
    Parses a string input (comma-separated, newline-separated) into a NumPy array.
    """
    try:
        # Each row is terminated by a new line char (user inputted) so use this for the split function
        rows = input_str.strip().split('\n')

        matrix = []

        # In each row, split the contents further by ',' char as this seperates the elements in the row
        for row in rows:
            elements = [float(e.strip()) for e in row.strip().split(',') if e.strip()]

            # If we have elements filled up, then add a row in our python list
            if elements:
                matrix.append(elements)

        # Loop done and nothing in the list, return the state from the function
        if not matrix:
            return None, "Matrix is empty."

        # Convert to NumPy array
        np_matrix = np.array(matrix)

        return np_matrix, None

    except ValueError:
        return None, "Invalid input. Please ensure all elements are valid numbers and formatted correctly."
    except Exception as e:
        return None, f"An error occurred during parsing: {e}"

# Shows the main title of the page

# Inject custom CSS to set the width of the sidebar
st.set_page_config(layout="wide") # Optional: use wide mode for more horizontal space

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem; /* Adjust the top padding (default is 3rem or 6rem) */
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 300px !important;  # Set the width to your desired value (e.g., 400px)
    }
    </style>
    """, unsafe_allow_html=True,)

st.sidebar.subheader("Choose the matrix operation")
operation = st.sidebar.selectbox("Operation", ("Do Multiplication", "Solve algebraic expressions", "Find Eigen values and vector"))

# This section manages the multiplication page
if operation == "Do Multiplication":
    st.title("Matrix multiplication app")

    st.markdown("**Significance:**")
    
    st.markdown("""
                Matrix multiplication is crucial because it efficiently handles complex linear transformations (rotations, scaling) 
                in graphics/physics, powers AI/ML models for pattern recognition (deep learning uses it for neurons), solves systems 
                of equations, models real-world data (images, language), and represents compositions of operations, making it a fundamental tool
                across science, engineering, and tech.""")

    # Puts 2 new empty lines
    st.write("\n")

    st.write("Enter matrices below using comma-separated values for elements in a row. Press enter key to go to the next row.")

    # Create 2 cols so that we can divide the page vertically
    col1, col2 = st.columns(2)

    with col1:
        st.write("Matrix A")
        input_a = st.text_area("Enter Matrix A values:", "1, 2\n3, 4", height=125, key="matrix_a_input")

    with col2:
        st.write("Matrix B")
        input_b = st.text_area("Enter Matrix B values:", "5, 6\n7, 8", height=125, key="matrix_b_input")

    # Handle the output - by directly calling out what needs to happen when a button labbeld Multiply...is clicked
    if st.button("Multiply Matrices"):
        matrix_a, error_a = parse_matrix(input_a)
        matrix_b, error_b = parse_matrix(input_b)

        if error_a:
            st.error(f"Error in Matrix A: {error_a}")
        elif error_b:
            st.error(f"Error in Matrix B: {error_b}")
        elif matrix_a is not None and matrix_b is not None:
            # 3 cols introduced to show the result and strings in a more compact way
            col_inner_1, col_inner_2, col_inner_3 = st.columns(3)

            try:
                # Perform matrix multiplication using numpy.dot or the @ operator
                result = np.dot(matrix_a, matrix_b)

                with col_inner_1:
                    # Show dimensions and confirmation message
                    st.success(
                    f"Multiplication successful. Resulting matrix has dimensions {result.shape[0]}x{result.shape[1]}.")
                with col_inner_2:
                    # Display result using a Pandas DataFrame for better formatting
                    st.dataframe(pd.DataFrame(result))

            except ValueError as e:
                    # Errors are thrown in across a single row and col
                    st.error(f"Multiplication not possible: {e}")
                    
# This section manages the algebraic expressions solving
if operation == "Solve algebraic expressions":
    st.title("Solve Algebraic equations using matrix inverse")

    st.markdown("**Significance:**")

    st.markdown("""To solve two algebraic equations using the inverse method in Python, we will represent the system of linear equations 
                in the form **AX = B**, where the solution is found by **X=A^-1 B.**""")

    # Puts 2 new empty lines
    st.write("\n\n")

    st.write("Enter matrices below (A and B) using comma-separated values for elements and newlines for rows.")
   
    # Create 2 cols so that we can divide the page vertically
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Matrix A")
        input_a = st.text_area("Enter Matrix A values:", "3, 1\n1, 2", height=125, key="matrix_a_algex_input")
    with col2:
        st.write("Matrix X")
        st.text_area("Matrix X values:", "x\ny", height=125, disabled=True)
    with col3:
        st.write("Matrix B")
        input_b = st.text_area("Enter Matrix B values:", "9\n8", height=125, key="matrix_b_algexe_input")

    if st.button("Solve Equations"):
        matrix_a, error_a = parse_matrix(input_a)
        matrix_b, error_b = parse_matrix(input_b)

        if error_a:
            st.error(f"Error in Matrix A: {error_a}")
        elif error_b:
            st.error(f"Error in Matrix B: {error_b}")
        elif matrix_a is not None and matrix_b is not None:
            col_inner_1, col_inner_2, col_inner_3 = st.columns(3)

            try:
                det_A = np.linalg.det(matrix_a)

                if det_A == 0:
                    st.error("The matrix is singular and does not have an inverse. The system may have no unique solution.")
                else:
                    # 2. Calculate the inverse of matrix A (A_inv)
                    A_inv = np.linalg.inv(matrix_a)

                    with col_inner_1:
                        st.markdown("**Inverse of A**")

                        # Display result using a Pandas DataFrame for better formatting
                        st.dataframe(pd.DataFrame(A_inv))
                    with col_inner_2:
                        # 3. Calculate the solution vector X by multiplying A_inv and B
                        # The '@' operator performs matrix multiplication in NumPy
                        X = A_inv @ matrix_b
                        st.markdown("**Solution for X (x, y)**")

                        # First, flatten the numpy array to 1D and then convert to list
                        X_list = list(X.flatten())
                        X_value = int(X_list[0])
                        Y_value = int(X_list[1])

                        st.markdown(f"Value is x: **{X_value}** and y: **{Y_value}**")
                    with col_inner_3:
                        # Verification - Check if A @ X equals B
                        B_check = matrix_a @ X
                        st.markdown("**Verification:  (A * X) should be equal to B:**")
                        st.dataframe(pd.DataFrame(B_check))
            except np.linalg.LinAlgError as e:
                st.error(f"Error calculating inverse: {e}")

# This section manages the Eigen things page
if operation == "Find Eigen values and vector":
    st.title("Eigen values and vector app")

    st.markdown("**Significance:**")

    st.markdown("""In linear algebra, an eigen-vector or characteristic vector of a linear transformation is a non-zero vector that change
    at most by a constant factor when that linear transformation is applied to it. The corresponding eigen value, often represented by a lamda
    , is the multiplying factor. **The eigen-value is the factor by which an eigen vector is stretched (not rotated or sheared). 
    If the eigen value is negative, the direction is reversed.**""")

    # Puts 2 new empty lines
    st.write("\n\n")

    st.write("Enter a matrix below using comma-separated values for elements and newlines for rows.")

    col1, col2, col3 = st.columns([5, 1, 1])

    with col1:
        st.write("Matrix A")
        input_e = st.text_area("Enter Matrix A values:", "-6, 3\n4, 5", height=125, key="matrix_e_input")

        if st.button("Find Eigen values and vectors"):
            matrix_e, error_e = parse_matrix(input_e)

            if error_e:
                st.error(f"Error in Matrix A: {error_e}")
            elif matrix_e is not None:
                try:
                    # Use the formula to get the eigen things
                    w, v = np.linalg.eig(matrix_e)

                    # Divide the page further in 2 cols - to show both the values seperately
                    col_inner_1, col_inner_2, col_inner_3 = st.columns(3)

                    with col_inner_1:
                        # Show confirmation message
                        st.success(f"Computation of Eigen values and vectors successful.")

                    with col_inner_2:
                        st.markdown("**Eigen values**")

                        # Display result using a Pandas DataFrame for better formatting
                        st.dataframe(pd.DataFrame(w))
                    with col_inner_3:
                        st.markdown("**Eigen vector**")

                        # Display result using a Pandas DataFrame for better formatting
                        st.dataframe(pd.DataFrame(v))
                except ValueError as e:
                    st.error(f"Operation not possible: {e}")

