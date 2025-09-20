import streamlit as st
import pandas as pd
import numpy as np
from pyresparser import ResumeParser
import warnings
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os
from fuzzywuzzy import fuzz
import spacy
import PyPDF2

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Resume-Skills Matcher for Google Form Data",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_nlp_model():
    """Load spaCy model with error handling"""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("spaCy English model not found. Please install it using: python -m spacy download en_core_web_sm")
        return None

def extract_skills_from_resume(uploaded_file):
    """Extract skills from uploaded resume using pyresparser"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Parse resume
        data = ResumeParser(tmp_file_path).get_extracted_data()

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return data
    except Exception as e:
        st.error(f"Error parsing resume: {str(e)}")
        return None

def parse_multi_select_field(field_value):
    """Parse multi-select fields that may contain multiple values"""
    if pd.isna(field_value) or field_value == '':
        return []

    # Convert to string in case it's not
    field_str = str(field_value).strip()

    # Common separators for multi-select fields in Google Forms
    separators = [';', ',', '|', '\n', '\r\n', '\r', ' & ', ' and ', ' / ']

    # Split by multiple separators
    items = [field_str]
    for sep in separators:
        new_items = []
        for item in items:
            new_items.extend([x.strip() for x in item.split(sep) if x.strip()])
        items = new_items

    # Clean and filter items
    cleaned_items = []
    for item in items:
        item = item.strip()
        if len(item) > 1 and item.lower() not in ['na', 'n/a', 'none', 'nil']:
            cleaned_items.append(item.lower())

    return list(set(cleaned_items))  # Remove duplicates

def process_google_form_data(df):
    """Process Google Form responses with specific column structure"""
    employees_data = []

    # Define column mappings (handle variations in column names)
    column_mappings = {
        'name': ['Name', 'name', 'Full Name', 'Employee Name'],
        'email': ['Email Address', 'Email', 'email', 'Email ID'],
        'skills': ['Skill/Profile/Domain', 'Skills', 'Domain', 'Profile', 'Technical Skills'],
        'company': ['Company Name', 'Company', 'Organization'],
        'designation': ['Designation', 'Role', 'Position', 'Job Title'],
        'experience': ['Total No.of Years of Experience', 'Experience', 'Years of Experience'],
        'certifications': ['Certifications', 'Certification', 'Certificates'],
        'specialization': ['Specialization', 'Specialisation', 'Area of Expertise'],
        'mobile': ['Mobile Number', 'Phone', 'Contact Number', 'Mobile'],
        'linkedin': ['LinkedIn Profile', 'LinkedIn', 'LinkedIn URL'],
        'mandal': ['Mandal', 'Location', 'Region'],
        'address': ['Address', 'Location', 'City']
    }

    def find_column(mapping_key):
        """Find the actual column name based on possible variations"""
        possible_names = column_mappings.get(mapping_key, [])
        for col in df.columns:
            if col in possible_names:
                return col
        return None

    for index, row in df.iterrows():
        employee_info = {
            'index': index,
            'name': '',
            'email': '',
            'skills': [],
            'all_skills': [],  # Combined skills from multiple sources
            'company': '',
            'designation': '',
            'experience': '',
            'certifications': [],
            'specialization': [],
            'mobile': '',
            'linkedin': '',
            'mandal': '',
            'address': ''
        }

        # Extract basic information
        name_col = find_column('name')
        if name_col and pd.notna(row[name_col]):
            employee_info['name'] = str(row[name_col]).strip()

        email_col = find_column('email')
        if email_col and pd.notna(row[email_col]):
            employee_info['email'] = str(row[email_col]).strip()

        company_col = find_column('company')
        if company_col and pd.notna(row[company_col]):
            employee_info['company'] = str(row[company_col]).strip()

        designation_col = find_column('designation')
        if designation_col and pd.notna(row[designation_col]):
            employee_info['designation'] = str(row[designation_col]).strip()

        experience_col = find_column('experience')
        if experience_col and pd.notna(row[experience_col]):
            employee_info['experience'] = str(row[experience_col]).strip()

        mobile_col = find_column('mobile')
        if mobile_col and pd.notna(row[mobile_col]):
            employee_info['mobile'] = str(row[mobile_col]).strip()

        linkedin_col = find_column('linkedin')
        if linkedin_col and pd.notna(row[linkedin_col]):
            employee_info['linkedin'] = str(row[linkedin_col]).strip()

        mandal_col = find_column('mandal')
        if mandal_col and pd.notna(row[mandal_col]):
            employee_info['mandal'] = str(row[mandal_col]).strip()

        address_col = find_column('address')
        if address_col and pd.notna(row[address_col]):
            employee_info['address'] = str(row[address_col]).strip()

        # Extract skills from primary skills column
        skills_col = find_column('skills')
        if skills_col and pd.notna(row[skills_col]):
            employee_info['skills'] = parse_multi_select_field(row[skills_col])

        # Extract certifications
        cert_col = find_column('certifications')
        if cert_col and pd.notna(row[cert_col]):
            employee_info['certifications'] = parse_multi_select_field(row[cert_col])

        # Extract specializations
        spec_col = find_column('specialization')
        if spec_col and pd.notna(row[spec_col]):
            employee_info['specialization'] = parse_multi_select_field(row[spec_col])

        # Combine all skills from different sources
        all_skills = (employee_info['skills'] +
                     employee_info['certifications'] +
                     employee_info['specialization'])
        employee_info['all_skills'] = list(set(all_skills))  # Remove duplicates

        # Use all_skills as the main skills list for matching
        employee_info['skills'] = employee_info['all_skills']

        employees_data.append(employee_info)

    return employees_data

def calculate_skill_match_score(resume_skills, employee_skills):
    """Calculate matching score between resume skills and employee skills"""
    if not resume_skills or not employee_skills:
        return 0

    resume_skills_set = set([skill.lower().strip() for skill in resume_skills])
    employee_skills_set = set([skill.lower().strip() for skill in employee_skills])

    # Exact matches
    exact_matches = len(resume_skills_set.intersection(employee_skills_set))

    # Fuzzy matching for partial matches
    fuzzy_score = 0
    fuzzy_matches = []

    for resume_skill in resume_skills_set:
        best_match = ""
        best_similarity = 0

        for emp_skill in employee_skills_set:
            similarity = fuzz.ratio(resume_skill, emp_skill)
            if similarity > 70 and similarity > best_similarity:  # 70% similarity threshold
                best_similarity = similarity
                best_match = emp_skill

        if best_similarity > 70:
            fuzzy_score += best_similarity / 100
            fuzzy_matches.append(f"{resume_skill} ≈ {best_match} ({best_similarity}%)")

    # Combine exact and fuzzy matches
    total_skills = len(resume_skills_set)
    if total_skills == 0:
        return 0, [], []

    # Weighted score: 70% exact matches, 30% fuzzy matches
    final_score = (exact_matches * 0.7 + fuzzy_score * 0.3) / total_skills * 100

    exact_match_list = list(resume_skills_set.intersection(employee_skills_set))

    return min(final_score, 100), exact_match_list, fuzzy_matches

def find_top_matches(resume_skills, employees_data, top_n=5):
    """Find top N matching employees based on skills"""
    scored_employees = []

    for employee in employees_data:
        score, exact_matches, fuzzy_matches = calculate_skill_match_score(resume_skills, employee['skills'])
        employee['match_score'] = score
        employee['exact_matches'] = exact_matches
        employee['fuzzy_matches'] = fuzzy_matches

        scored_employees.append(employee)

    # Sort by match score descending
    scored_employees.sort(key=lambda x: x['match_score'], reverse=True)

    return scored_employees[:top_n]

def display_results(top_matches, resume_data):
    """Display matching results in a nice format"""
    st.subheader("🎯 Top 5 Referral Candidates")

    if not top_matches:
        st.warning("No matching candidates found.")
        return

    # Display resume information
    with st.expander("📄 Resume Analysis", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if resume_data and resume_data.get('name'):
                st.write(f"**Candidate Name:** {resume_data['name']}")
            if resume_data and resume_data.get('email'):
                st.write(f"**Email:** {resume_data['email']}")
            if resume_data and resume_data.get('total_experience'):
                st.write(f"**Experience:** {resume_data['total_experience']} years")

        with col2:
            if resume_data and resume_data.get('skills'):
                st.write("**Extracted Skills:**")
                skills_str = ', '.join(resume_data['skills'][:15])  # Show first 15 skills
                if len(resume_data['skills']) > 15:
                    skills_str += f" ... and {len(resume_data['skills']) - 15} more"
                st.write(skills_str)

    # Display top matches
    for i, match in enumerate(top_matches, 1):
        with st.container():
            st.markdown(f"### 🏆 Rank #{i}")

            # Create columns for layout
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                # Score visualization
                score_color = "green" if match['match_score'] >= 70 else "orange" if match['match_score'] >= 40 else "red"
                st.markdown(f"<div style='background-color: {score_color}; color: white; padding: 15px; border-radius: 10px; text-align: center;'>"
                          f"<strong>{match['match_score']:.1f}%</strong><br>Match Score</div>",
                          unsafe_allow_html=True)

                st.write(f"**Experience:** {match['experience']}")

            with col2:
                # Employee details
                st.write(f"**👤 Name:** {match['name']}")
                st.write(f"**📧 Email:** {match['email']}")
                st.write(f"**🏢 Company:** {match['company']}")
                st.write(f"**💼 Designation:** {match['designation']}")

                if match['mandal']:
                    st.write(f"**📍 Location:** {match['mandal']}")

                if match['linkedin']:
                    st.write(f"**🔗 LinkedIn:** {match['linkedin']}")

                # Exact matching skills
                if match['exact_matches']:
                    st.write("**🎯 Exact Skill Matches:**")
                    st.success(", ".join(match['exact_matches']))

                # Fuzzy matching skills
                if match['fuzzy_matches']:
                    st.write("**🔍 Similar Skills Found:**")
                    for fuzzy_match in match['fuzzy_matches'][:3]:  # Show top 3 fuzzy matches
                        st.info(fuzzy_match)

            with col3:
                # Additional skills and info
                st.write("**💡 All Skills:**")
                if len(match['skills']) > 8:
                    skills_preview = match['skills'][:8]
                    st.write(", ".join(skills_preview))
                    st.write(f"*... and {len(match['skills']) - 8} more*")
                else:
                    st.write(", ".join(match['skills']))

                if match['certifications']:
                    st.write("**🏆 Certifications:**")
                    cert_preview = match['certifications'][:3]
                    st.write(", ".join(cert_preview))
                    if len(match['certifications']) > 3:
                        st.write(f"*... and {len(match['certifications']) - 3} more*")

            st.divider()

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_skills_spacy(text, skill_keywords=None):
    nlp = load_nlp_model()
    doc = nlp(text)
    # Basic candidate skill extraction using keyword search and noun chunks.
    skill_set = set()
    # Pre-defined skill list (extend or load as needed)
    if skill_keywords is None:
        skill_keywords = [
            "java", "python", "aws", "nodejs", "spring", "docker", "react", "angular", "postgresql", "cloud", "machine learning",
            "django", "kubernetes", "devops", "sql", "data science", "nlp", "mongodb", "redis", "testing", "selenium",
            "backend", "frontend", "network", "security", "management", "product management", "program management", "azure", "gcp",
            "database"
            # Add more as needed
        ]
    skill_keywords = set([kw.lower() for kw in skill_keywords])
    # Pull noun chunks and match against keywords
    for chunk in doc.noun_chunks:
        term = chunk.text.lower().strip()
        for kw in skill_keywords:
            if kw in term:
                skill_set.add(kw)
    # Direct keyword match from the whole text
    for kw in skill_keywords:
        if kw in text.lower():
            skill_set.add(kw)
    return list(skill_set)

def extract_skills_from_resume_spacy(uploaded_file):
    text = extract_text_from_pdf(uploaded_file)
    skills = extract_skills_spacy(text)
    return {
        "skills": skills
    }

def main():
    st.title("🎯 Resume-Skills Matcher for Google Form Data")
    st.markdown("Upload your Google Form responses Excel file and a candidate's resume to find the best 5 referral matches!")

    # Sidebar for instructions
    with st.sidebar:
        st.header("📋 Expected Columns")
        st.markdown("""
        Your Google Form Excel should have:
        - **Name**
        - **Email Address**
        - **Skill/Profile/Domain** (multi-select)
        - **Company Name**
        - **Designation**
        - **Total No.of Years of Experience**
        - **Certifications** (multi-select)
        - **Specialization** (multi-select)
        - **Mandal** (Location)
        - **LinkedIn Profile**
        """)

        st.header("📊 Multi-Select Support")
        st.markdown("""
        The app handles multi-select fields with separators:
        - Semicolons (;)
        - Commas (,)
        - Pipes (|)
        - Line breaks
        - "and", "&", "/"
        """)

        st.header("🎯 Matching Logic")
        st.markdown("""
        - Combines skills from Skills, Certifications, and Specialization
        - Exact matches (70% weight)
        - Fuzzy/similar matches (30% weight)
        - Threshold: 70%+ similarity
        """)

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Upload Google Form Data")

        # File uploader for Google Form data
        form_file = st.file_uploader(
            "Choose Excel or CSV file with Google Form responses",
            type=['xlsx', 'xls', 'csv'],
            help="Upload your Google Form responses containing employee skills data"
        )

        employees_df = None
        employees_data = None

        if form_file is not None:
            try:
                # Read the file
                if form_file.name.endswith('.csv'):
                    employees_df = pd.read_csv(form_file)
                else:
                    employees_df = pd.read_excel(form_file)

                st.success(f"✅ Loaded {len(employees_df)} employee records")

                # Show preview
                with st.expander("Preview Data", expanded=False):
                    st.dataframe(employees_df.head())

                # Show detected columns
                with st.expander("Detected Columns", expanded=False):
                    st.write("**Available columns:**")
                    for col in employees_df.columns:
                        st.write(f"- {col}")

                # Process the data automatically
                with st.spinner("Processing Google Form data..."):
                    employees_data = process_google_form_data(employees_df)

                st.info(f"Processed {len(employees_data)} employee profiles")

                # Show processing summary
                with st.expander("Processing Summary", expanded=False):
                    total_skills = sum(len(emp['skills']) for emp in employees_data)
                    avg_skills = total_skills / len(employees_data) if employees_data else 0
                    st.write(f"**Total skills extracted:** {total_skills}")
                    st.write(f"**Average skills per employee:** {avg_skills:.1f}")

                    # Show sample processed employee
                    if employees_data:
                        sample_emp = employees_data[0]
                        st.write("**Sample processed employee:**")
                        st.json({
                            "name": sample_emp['name'],
                            "company": sample_emp['company'],
                            "skills_count": len(sample_emp['skills']),
                            "sample_skills": sample_emp['skills'][:5]
                        })

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    with col2:
        st.subheader("📄 Upload Resume")

        # File uploader for resume
        resume_file = st.file_uploader(
            "Choose resume file (PDF)",
            type=['pdf'],
            help="Upload candidate's resume in PDF format"
        )

        resume_data = None

        if resume_file is not None:
            with st.spinner("Parsing resume..."):
                resume_data = extract_skills_from_resume_spacy(resume_file)

            if resume_data:
                st.success("✅ Resume parsed successfully")

                # Show extracted information
                with st.expander("Extracted Resume Data", expanded=False):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        if resume_data.get('name'):
                            st.write(f"**Name:** {resume_data['name']}")
                        if resume_data.get('email'):
                            st.write(f"**Email:** {resume_data['email']}")
                        if resume_data.get('total_experience'):
                            st.write(f"**Experience:** {resume_data['total_experience']} years")
                        if resume_data.get('degree'):
                            st.write(f"**Degree:** {', '.join(resume_data['degree'])}")

                    with col_b:
                        if resume_data.get('skills'):
                            st.write(f"**Skills Found:** {len(resume_data['skills'])}")
                            skills_display = ", ".join(resume_data['skills'][:15])
                            if len(resume_data['skills']) > 15:
                                skills_display += f" ... and {len(resume_data['skills']) - 15} more"
                            st.write(skills_display)

    # Process and show results
    if employees_data and resume_data and resume_data.get('skills'):
        st.markdown("---")

        if st.button("🔍 Find Best Matches", type="primary", use_container_width=True):
            with st.spinner("Finding best matches..."):
                # Find top matches
                top_matches = find_top_matches(resume_data['skills'], employees_data)

                # Display results
                display_results(top_matches, resume_data)

                # Create downloadable CSV of results
                if top_matches:
                    st.markdown("---")

                    # Prepare data for CSV
                    csv_data = []
                    for i, match in enumerate(top_matches, 1):
                        csv_data.append({
                            'Rank': i,
                            'Name': match['name'],
                            'Email': match['email'],
                            'Company': match['company'],
                            'Designation': match['designation'],
                            'Experience': match['experience'],
                            'Match Score (%)': round(match['match_score'], 1),
                            'Exact Matches': ', '.join(match['exact_matches']),
                            'All Skills': ', '.join(match['skills'][:20]),  # Top 20 skills
                            'LinkedIn': match['linkedin'],
                            'Location': match['mandal']
                        })

                    results_df = pd.DataFrame(csv_data)

                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"referral_matches_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    elif employees_data and resume_data:
        if not resume_data.get('skills'):
            st.warning("⚠️ No skills found in the uploaded resume. Please check if the resume contains skill information.")
    elif employees_df is not None and not employees_data:
        st.warning("⚠️ Could not process Google Form data. Please check if your file has the expected columns.")

    # Footer
    st.markdown("---")
    st.markdown("""
    **💡 Tips:**
    - Ensure your Google Form has multi-select questions for skills
    - Use clear skill names (e.g., "Java", "Python", "AWS")
    - Include certifications and specializations for better matching
    - Resume should have a clear skills section
    """)

if __name__ == "__main__":
    main()%
 n0p00b8@m-cp4ch00jh3  ~/Downloads/ResumeMatchingForReferrals  ↱ main  ls
google_form_resume_matcher.py README.md
 n0p00b8@m-cp4ch00jh3  ~/Downloads/ResumeMatchingForReferrals  ↱ main  cd ..
 n0p00b8@m-cp4ch00jh3  ~/Downloads  cd ResumeMatching
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os
from fuzzywuzzy import fuzz
import spacy
import PyPDF2

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Resume-Skills Matcher for Google Form Data",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_nlp_model():
    """Load spaCy model with error handling"""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("spaCy English model not found. Please install it using: python -m spacy download en_core_web_sm")
        return None

def parse_multi_select_field(field_value):
    """Parse multi-select fields that may contain multiple values"""
    if pd.isna(field_value) or field_value == '':
        return []

    # Convert to string in case it's not
    field_str = str(field_value).strip()

    # Common separators for multi-select fields in Google Forms
    separators = [';', ',', '|', '\n', '\r\n', '\r', ' & ', ' and ', ' / ']

    # Split by multiple separators
    items = [field_str]
    for sep in separators:
        new_items = []
        for item in items:
            new_items.extend([x.strip() for x in item.split(sep) if x.strip()])
        items = new_items

    # Clean and filter items
    cleaned_items = []
    for item in items:
        item = item.strip()
        if len(item) > 1 and item.lower() not in ['na', 'n/a', 'none', 'nil']:
            cleaned_items.append(item.lower())

    return list(set(cleaned_items))  # Remove duplicates

"google_form_resume_matcher.py" [noeol] 565L, 23183B
