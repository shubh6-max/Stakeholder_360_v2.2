import streamlit as st

def set_common_page_config(page_title: str, layout: str = "wide"):
    """
    Apply a shared favicon and layout settings across all Streamlit pages.
    Must be called before any other Streamlit command.

    Parameters
    ----------
    title : str
        Title of the Streamlit page (appears in browser tab)
    layout : str, optional
        Page layout mode ('centered' or 'wide'), by default 'wide'
    """
    st.set_page_config(
        page_title=page_title,
        page_icon="https://media.licdn.com/dms/image/v2/D4D0BAQFSLuRei6pVZA/company-logo_200_200/B4DZfJuCNfGgAI-/0/1751435978200/themathcompany_logo?e=1762992000&v=beta&t=MSRoOFnBWF_-ppGgieJzqOsYaGH53_IqO__xbc_H7XY",
        layout=layout,
    )
