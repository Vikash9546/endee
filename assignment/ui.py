
import streamlit as st
import os

def apply_custom_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {
            --primary: #4F46E5;
            --primary-glow: rgba(79, 70, 229, 0.1);
            --bg-light: #FFFFFF;
            --bg-dark: #0B0F19;
            --card-light: #FFFFFF;
            --card-dark: rgba(30, 41, 59, 0.7);
            --sidebar-light: #F9FAFB;
            --sidebar-dark: #0F172A;
            --text-light: #111827;
            --text-dark: #F8FAFC;
            --border-light: #E5E7EB;
            --border-dark: rgba(51, 65, 85, 0.5);
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        }

        /* Base Application Styles */
        .stApp {
            background-color: var(--bg-light) !important;
            background-image: radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.03) 0, transparent 50%), 
                              radial-gradient(at 50% 0%, rgba(79, 70, 229, 0.05) 0, transparent 50%) !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        [data-testid="stAppViewContainer"] {
            padding: 0 !important;
        }

        /* Sidebar Glassmorphism */
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-light) !important;
            border-right: 1px solid var(--border-light) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.02);
            padding-top: 2rem;
        }

        /* Sidebar Navigation Cleanup */
        [data-testid="stSidebarNav"] { display: none !important; }
        
        /* Sidebar Navigation Buttons Overwrite */
        .stButton > button {
            border: none !important;
            background-color: transparent !important;
            color: #64748B !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.75rem 1.25rem !important;
            border-radius: 12px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            margin-bottom: 2px !important;
        }

        .stButton > button:hover {
            background-color: #F1F5F9 !important;
            color: var(--primary) !important;
            transform: translateX(4px);
        }

        /* Brand Styling */
        .brand-container {
            padding: 0 1.5rem 2.5rem 1.5rem;
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-name {
            font-weight: 800;
            font-size: 1.4rem;
            color: #111827;
            margin: 0;
            letter-spacing: -0.01em;
        }

        .brand-tagline {
            font-size: 0.75rem;
            color: #94A3B8;
            margin: 0;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Hero */
        .hero-container {
            text-align: center;
            padding: 6rem 1rem 4rem 1rem;
            max-width: 900px;
            margin: 0 auto;
        }

        .hero-title {
            font-size: 4rem;
            font-weight: 800;
            color: var(--text-light);
            margin-bottom: 1.5rem;
            letter-spacing: -0.04em;
            line-height: 1.1;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: #4B5563;
            line-height: 1.6;
            margin-bottom: 3.5rem;
        }

        /* Premium White Cards */
        .card-container {
            display: flex;
            gap: 24px;
            margin-bottom: 5rem;
            justify-content: center;
            flex-wrap: wrap;
            padding: 0 2rem;
        }

        .feature-card {
            background: var(--card-light);
            padding: 2.5rem;
            border-radius: 24px;
            border: 1px solid var(--border-light);
            flex: 1;
            min-width: 280px;
            max-width: 320px;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }

        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: var(--shadow-lg);
            border-color: var(--primary);
        }

        .icon-box {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.75rem;
            font-size: 1.5rem;
        }

        .card-title {
            font-size: 1.4rem;
            font-weight: 800;
            color: var(--text-light);
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        }

        .card-desc {
            font-size: 0.95rem;
            color: #6B7280;
            line-height: 1.6;
        }

        /* Quote */
        .quote-section {
            background: #F5F7FF;
            padding: 3.5rem;
            border-radius: 32px;
            text-align: center;
            max-width: 950px;
            margin: 0 auto 5rem auto;
            border: 1px solid rgba(79, 70, 229, 0.08);
        }

        .quote-text {
            font-size: 1.8rem;
            font-weight: 600;
            line-height: 1.4;
            color: #1E1B4B;
            margin-bottom: 2rem;
            letter-spacing: -0.01em;
        }

        .quote-author {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            color: var(--primary);
        }

        /* Dark Mode Override */
        @media (prefers-color-scheme: dark) {
            .stApp {
                background: linear-gradient(135deg, var(--bg-dark) 0%, #111827 100%) !important;
            }
            [data-testid="stSidebar"] {
                background-color: var(--sidebar-dark) !important;
                border-color: var(--border-dark) !important;
            }
            .feature-card {
                background: var(--card-dark);
                border-color: var(--border-dark);
                box-shadow: none;
            }
            .card-title, .hero-title, .brand-name {
                color: var(--text-dark) !important;
            }
            .hero-subtitle, .card-desc, .brand-tagline {
                color: #9CA3AF;
            }
            .quote-section {
                background: rgba(30, 58, 138, 0.1);
                border-color: rgba(30, 58, 138, 0.2);
            }
            .quote-text { color: #BFDBFE; }
            .stButton > button:hover {
                background-color: rgba(255, 255, 255, 0.05) !important;
            }
            .stButton > button { color: #9CA3AF !important; }
            
        }


        .stat-card {
            background: white !important;
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid var(--border-light);
            box-shadow: var(--shadow-sm);
        }

        @media (prefers-color-scheme: dark) {
            .stat-card {
                background: var(--card-dark) !important;
                border-color: var(--border-dark);
                box-shadow: none;
            }
            .stat-label { color: #9CA3AF !important; }
            .stat-value { color: var(--text-dark) !important; }
        }

        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    """, unsafe_allow_html=True)

def render_sidebar():
    # Use relative paths for cloud compatibility
    logo_path = "assignment/assets/logo.png"

    with st.sidebar:
        st.markdown(f"""
            <div class="brand-container">
                <img src="data:image/png;base64,{get_base64(logo_path)}" width="40" style="border-radius: 8px;">
                <div>
                    <h3 class="brand-name">Curator AI</h3>
                    <p class="brand-tagline">Knowledge Engine</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        pages = {
            "Dashboard": "fa-th-large",
            "Uploads": "fa-cloud-upload",
            "Library": "fa-folder",
            "Analytics": "fa-chart-line",
            "Settings": "fa-cog"
        }
        
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Dashboard"
            
        for page, icon in pages.items():
            is_active = "active" if st.session_state.current_page == page else ""
            if st.button(f"{page}", key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
                st.rerun()
                
        # Existing Incident Agent (Ghost Protocol) hidden or separate feature
        st.markdown("<br><hr><br>", unsafe_allow_html=True)


import base64
def get_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
