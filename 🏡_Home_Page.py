import streamlit as st

def set_page_configuration():
    st.set_page_config(page_title="Hello", page_icon="🏡")

def main():

    page_bg_img = """
                    <style>
                    [data-testid="stAppViewContainer"] {
                        background-image: url("https://i.pinimg.com/originals/b3/56/7b/b3567bdb76c15bda852fad6c08a8d8f2.png");

                        background-size: 100% 100%;
                    }
                    [data-testid="stHeader"]{
                        background: rgba(0,0,0,0);
                    }
                    [data-testid="stToolbar"]{
                        right:2rem;
                    }
                    [data-testid="stSidebar"] > div:first-child {
                        background-position: center;
                        background-color: rgba(129, 110, 95, 0.8);  

                    }
                    </style>
                    """
    st.markdown(page_bg_img,unsafe_allow_html=True)

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Write project introduction
    st.write("# 🏡 My Project Introduction 📸")

    st.markdown(
        """
        ### Project Information \n
        - Project description: Project is built in Python on Streamlit. \n
        - Course name: Digital Image Processing - DIPR430685_23_1_01 \n
        - Supervisor : MSc. Trần Tiến Đức \n
        - Performer: Nguyễn Thị Tường Vi - 20133113 \n
        ### Content \n
        1. Quadratic Equation Solver \n
        2. Facial Recognition \n
        3. Object Recognition \n
        4. Handwritten Digit Recognition \n
        5. Fruit Recognition \n
        6. Image Process \n
        7. Distance estimation \n
        """
    )

    # Close the main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
