# # import streamlit as st
# # import pandas as pd
# # from st_aggrid import AgGrid, GridOptionsBuilder

# # df = pd.DataFrame({
# #     "Name": ["Alice", "Bob", "Charlie"],
# #     "Age": [25, 30, 35],
# #     "Role": ["Engineer", "Designer", "Manager"]
# # })

# # gb = GridOptionsBuilder.from_dataframe(df)
# # gb.configure_default_column(editable=True)  # Make all columns editable
# # gb.configure_grid_options(enableRangeSelection=True)

# # grid_options = gb.build()

# # grid_response = AgGrid(
# #     df,
# #     gridOptions=grid_options,
# #     update_mode="MODEL_CHANGED",
# #     fit_columns_on_grid_load=True
# # )

# # edited_df = grid_response["data"]

# # st.write("Edited Data:")
# # st.write(edited_df)


# # import streamlit as st
# # import pandas as pd

# # # Sample data
# # df = pd.DataFrame({
# #     "Name": ["Shubham", "Bob", "Charlie"],
# #     "Age": [25, 30, 35],
# #     "Role": ["Engineer", "Designer", "Manager"]
# # })

# # # Editable table (only one table shown)
# # edited_df = st.data_editor(
# #     df,
# #     num_rows="dynamic",          # allows add/remove rows
# #     use_container_width=True,
# #     key="editable_table"
# # )

# # # If you need the updated data internally, just use edited_df
# # # e.g., save to DB or use in logic:
# # if st.button("Save Changes"):
# #     st.success("Changes saved!")
# #     # here you could do: save_to_db(edited_df)


import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

# Sample DataFrame
df = pd.DataFrame({
    "Name": ["Shubham", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Role": ["Engineer", "Designer", "Manager"],
    "Active": [True, False, True]
})

# --- Configure Grid Options ---
gb = GridOptionsBuilder.from_dataframe(df)

# Make all columns editable
gb.configure_default_column(editable=True, sortable=True, filter=True, resizable=True)

# Add dropdown for Role column
gb.configure_column(
    "Role",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": ["Engineer", "Designer", "Manager", "Analyst"]},
)

# Add checkbox for Active column
gb.configure_column("Active", editable=True, cellEditor="agCheckboxCellEditor")

# Add custom cell styling (Age > 30 in red)
cell_style_jscode = JsCode("""
function(params) {
    if (params.value > 30) {
        return {'color': 'red', 'fontWeight': 'bold'};
    }
}
""")
gb.configure_column("Age", cellStyle=cell_style_jscode)

# Enable features
gb.configure_grid_options(
    enableRangeSelection=True,
    rowSelection="multiple",   # allow multi-row selection
    pagination=True,
    paginationPageSize=5,
)

grid_options = gb.build()

# --- Render AgGrid ---
grid_response = AgGrid(
    df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    fit_columns_on_grid_load=True,
    theme="alpine",   # try: "streamlit", "dark", "fresh", "material"
    allow_unsafe_jscode=True,
    height=400,
    reload_data=False,
)

# Get edited data
edited_df = grid_response["data"]
selected_rows = grid_response["selected_rows"]

# --- Interactivity ---
st.subheader("Edited Data")
st.dataframe(edited_df)

st.subheader("Selected Rows")
st.write(selected_rows)

# Save button
if st.button("💾 Save to CSV"):
    # edited_df.to_csv("edited_table.csv", index=False)
    st.success("Saved edited data to `edited_table.csv`")

import streamlit as st

# Inject your CSS
st.markdown("""
<style>
.form-container {
  width: 320px;
  border-radius: 0.75rem;
  background-color: rgba(17, 24, 39, 1);
  padding: 2rem;
  color: rgba(243, 244, 246, 1);
}

.title {
  text-align: center;
  font-size: 1.5rem;
  line-height: 2rem;
  font-weight: 700;
}

.form {
  margin-top: 1.5rem;
}

.input-group {
  margin-top: 0.25rem;
  font-size: 0.875rem;
  line-height: 1.25rem;
}

.input-group label {
  display: block;
  color: rgba(156, 163, 175, 1);
  margin-bottom: 4px;
}

.input-group input {
  width: 100%;
  border-radius: 0.375rem;
  border: 1px solid rgba(55, 65, 81, 1);
  outline: 0;
  background-color: rgba(17, 24, 39, 1);
  padding: 0.75rem 1rem;
  color: rgba(243, 244, 246, 1);
}

.input-group input:focus {
  border-color: rgba(167, 139, 250, 1);
}

.forgot {
  display: flex;
  justify-content: flex-end;
  font-size: 0.75rem;
  line-height: 1rem;
  color: rgba(156, 163, 175,1);
  margin: 8px 0 14px 0;
}

.forgot a,.signup a {
  color: rgba(243, 244, 246, 1);
  text-decoration: none;
  font-size: 14px;
}

.forgot a:hover, .signup a:hover {
  text-decoration: underline rgba(167, 139, 250, 1);
}

.sign {
  display: block;
  width: 100%;
  background-color: rgba(167, 139, 250, 1);
  padding: 0.75rem;
  text-align: center;
  color: rgba(17, 24, 39, 1);
  border: none;
  border-radius: 0.375rem;
  font-weight: 600;
}

.social-message {
  display: flex;
  align-items: center;
  padding-top: 1rem;
}

.line {
  height: 1px;
  flex: 1 1 0%;
  background-color: rgba(55, 65, 81, 1);
}

.social-message .message {
  padding-left: 0.75rem;
  padding-right: 0.75rem;
  font-size: 0.875rem;
  line-height: 1.25rem;
  color: rgba(156, 163, 175, 1);
}

.social-icons {
  display: flex;
  justify-content: center;
}

.social-icons .icon {
  border-radius: 0.125rem;
  padding: 0.75rem;
  border: none;
  background-color: transparent;
  margin-left: 8px;
}

.social-icons .icon svg {
  height: 1.25rem;
  width: 1.25rem;
  fill: #fff;
}

.signup {
  text-align: center;
  font-size: 0.75rem;
  line-height: 1rem;
  color: rgba(156, 163, 175, 1);
}
</style>
""", unsafe_allow_html=True)

# Inject HTML that uses these classes
st.markdown("""
<div class="form-container">
  <div class="title">Sign In</div>
  <form class="form">
    <div class="input-group">
      <label>Email</label>
      <input type="text" placeholder="Enter your email"/>
    </div>
    <div class="input-group">
      <label>Password</label>
      <input type="password" placeholder="Enter your password"/>
    </div>
    <div class="forgot"><a href="#">Forgot Password?</a></div>
    <button class="sign">Sign In</button>
    <div class="signup">Don’t have an account? <a href="#">Sign up</a></div>
  </form>
</div>
""", unsafe_allow_html=True)
