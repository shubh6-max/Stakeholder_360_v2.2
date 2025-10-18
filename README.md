<p align="center">
    <img src="https://img.icons8.com/external-tal-revivo-regular-tal-revivo/96/external-readme-is-a-easy-to-build-a-developer-hub-that-adapts-to-the-user-logo-regular-tal-revivo.png" align="center" width="30%">
</p>
<p align="center"><h1 align="center">STAKEHOLDER 360</h1></p>
<p align="center">
	<em><code>❯ **Stakeholder 360 v2.2** is a full-stack, AI-assisted platform for **stakeholder analysis**, **KPI alignment**, and **RAG (Retrieval-Augmented Generation)** on company and persona documents.  
It provides a modern, multi-page Streamlit UI, integrates **Azure OpenAI** for generation + embeddings, and uses **PostgreSQL** (with vector storage) for persistence.  
Typical workflows include:
- Curating stakeholders and personas, editing profiles, and visualizing org charts.
- Ingesting annual reports / notes, chunk-embedding them, and running contextual Q&A.
- Generating insight summaries and KPI mappings per stakeholder.</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/shubh6-max/Stakeholder_360_v2.2?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/shubh6-max/Stakeholder_360_v2.2?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/shubh6-max/Stakeholder_360_v2.2?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/shubh6-max/Stakeholder_360_v2.2?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

<code>❯ REPLACE-ME</code>

---

##  Features

<code>❯ REPLACE-ME</code>

---

##  Project Structure

```sh
└── Stakeholder_360_v2.2/
    ├── .github
    │   └── workflows
    ├── app.py
    ├── asset
    │   └── styles.css
    ├── components
    │   ├── __pycache__
    │   ├── aggrid_sections_all.py
    │   ├── cards.py
    │   ├── details.py
    │   ├── header.py
    │   ├── insights_embed.py
    │   ├── insights_view.py
    │   ├── kpi_view.py
    │   ├── profile_editor.py
    │   └── sections_readonly.py
    ├── features
    │   ├── insights
    │   ├── orgchart
    │   └── stakeholders
    ├── pages
    │   ├── edit_profile.py
    │   ├── full_org_chart.py
    │   ├── insights.py
    │   ├── login.py
    │   ├── main_app.py
    │   └── signup.py
    ├── requirements.txt
    ├── scripts
    │   └── daily_kpi_cache.py
    └── utils
        ├── __init__.py
        ├── __pycache__
        ├── auth.py
        ├── db.py
        ├── layout.py
        ├── page_config.py
        └── rag_db.py
```


###  Project Index
<details open>
	<summary><b><code>STAKEHOLDER_360_V2.2/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/app.py'>app.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- features Submodule -->
		<summary><b>features</b></summary>
		<blockquote>
			<details>
				<summary><b>insights</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/store.py'>store.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/ingest_embeddings.py'>ingest_embeddings.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/retrieve.py'>retrieve.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/config.py'>config.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/batch_core.py'>batch_core.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/vectorstore.py'>vectorstore.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/chunk_embed.py'>chunk_embed.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/clients.py'>clients.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/pipeline.py'>pipeline.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/fetchers.py'>fetchers.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/prompts.py'>prompts.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/ingest.py'>ingest.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/chains.py'>chains.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/chunk_embed_langchain.py'>chunk_embed_langchain.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/insights/fetch_annual.py'>fetch_annual.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>stakeholders</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/stakeholders/service.py'>service.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/stakeholders/models.py'>models.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>orgchart</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/orgchart/renderer.py'>renderer.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/features/orgchart/builder.py'>builder.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- .github Submodule -->
		<summary><b>.github</b></summary>
		<blockquote>
			<details>
				<summary><b>workflows</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/.github/workflows/daily-kpi-cache.yml'>daily-kpi-cache.yml</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- scripts Submodule -->
		<summary><b>scripts</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/scripts/daily_kpi_cache.py'>daily_kpi_cache.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- components Submodule -->
		<summary><b>components</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/components/header.py'>header.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/components/profile_editor.py'>profile_editor.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/components/cards.py'>cards.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/components/details.py'>details.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/components/sections_readonly.py'>sections_readonly.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/components/aggrid_sections_all.py'>aggrid_sections_all.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/components/kpi_view.py'>kpi_view.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/components/insights_embed.py'>insights_embed.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/components/insights_view.py'>insights_view.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- pages Submodule -->
		<summary><b>pages</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/pages/full_org_chart.py'>full_org_chart.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/pages/signup.py'>signup.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/pages/main_app.py'>main_app.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/pages/edit_profile.py'>edit_profile.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/pages/insights.py'>insights.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/pages/login.py'>login.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- asset Submodule -->
		<summary><b>asset</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/asset/styles.css'>styles.css</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- utils Submodule -->
		<summary><b>utils</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/utils/page_config.py'>page_config.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/utils/rag_db.py'>rag_db.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/utils/auth.py'>auth.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/utils/db.py'>db.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/master/utils/layout.py'>layout.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with Stakeholder_360_v2.2, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


###  Installation

Install Stakeholder_360_v2.2 using one of the following methods:

**Build from source:**

1. Clone the Stakeholder_360_v2.2 repository:
```sh
❯ git clone https://github.com/shubh6-max/Stakeholder_360_v2.2
```

2. Navigate to the project directory:
```sh
❯ cd Stakeholder_360_v2.2
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```




###  Usage
Run Stakeholder_360_v2.2 using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```


###  Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pytest
```


---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/shubh6-max/Stakeholder_360_v2.2/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/shubh6-max/Stakeholder_360_v2.2/issues)**: Submit bugs found or log feature requests for the `Stakeholder_360_v2.2` project.
- **💡 [Submit Pull Requests](https://github.com/shubh6-max/Stakeholder_360_v2.2/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/shubh6-max/Stakeholder_360_v2.2
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/shubh6-max/Stakeholder_360_v2.2/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=shubh6-max/Stakeholder_360_v2.2">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---