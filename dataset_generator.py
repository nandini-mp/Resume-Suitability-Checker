import pandas as pd
import random

# Create more comprehensive sample data
roles = ["Data Scientist", "Web Developer", "Software Engineer", "AI/ML Engineer"]

# Key skills for each role (used to generate realistic entries)
role_skills = {
    "Data Scientist": [
        # Core skills (high relevance)
        "Python", "R", "SQL", "Machine Learning", "Data Analysis", "Statistics",
        "Pandas", "NumPy", "Data Visualization", "Tableau", "Power BI",
        "Scikit-learn", "TensorFlow", "PyTorch", "Feature Engineering",
        # Related skills (medium relevance)
        "Big Data", "Data Mining", "A/B Testing", "Hypothesis Testing",
        "Data Warehousing", "ETL", "Data Modeling", "Statistical Analysis",
        # Background skills (lower relevance)
        "Mathematics", "Computer Science", "Research", "Critical Thinking"
    ],

    "Web Developer": [
        # Core skills (high relevance)
        "HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js",
        "Frontend Development", "Backend Development", "Responsive Design",
        "Web Design", "UI/UX", "Bootstrap", "jQuery", "REST APIs",
        # Related skills (medium relevance)
        "PHP", "MySQL", "MongoDB", "Express.js", "Django", "Flask", "Ruby on Rails",
        "WordPress", "GraphQL", "AJAX", "Webpack", "Git", "DevOps",
        # Background skills (lower relevance)
        "Computer Science", "Problem Solving", "User Experience"
    ],

    "Software Engineer": [
        # Core skills (high relevance)
        "Java", "C++", "C#", "Python", "JavaScript", "Algorithms", "Data Structures",
        "Object-Oriented Programming", "Software Development", "Version Control",
        "Git", "Unit Testing", "Debugging", "Software Architecture", "Agile",
        # Related skills (medium relevance)
        ".NET", "Spring", "Microservices", "Design Patterns", "CI/CD",
        "Docker", "Kubernetes", "Jenkins", "System Design", "Database Design",
        # Background skills (lower relevance)
        "Computer Science", "Problem Solving", "Documentation", "Communication"
    ],

    "AI/ML Engineer": [
        # Core skills (high relevance)
        "Neural Networks", "Deep Learning", "TensorFlow", "PyTorch", "Keras",
        "Machine Learning", "Model Training", "Computer Vision", "NLP",
        "Reinforcement Learning", "Python", "Feature Engineering", "Scikit-learn",
        # Related skills (medium relevance)
        "Data Preprocessing", "CUDA", "GPU Acceleration", "Hyperparameter Tuning",
        "Model Deployment", "MLOps", "Model Optimization", "Transfer Learning",
        # Background skills (lower relevance)
        "Mathematics", "Statistics", "Research", "Algorithms", "Data Structures"
    ]
}

# Non-matching skills (for generating less relevant resumes)
non_matching_skills = {
    "Data Scientist": ["Sales", "Marketing", "Customer Service", "Project Management",
                      "Accounting", "HR Management", "Content Writing"],
    "Web Developer": ["Data Analysis", "Financial Modeling", "Mechanical Engineering",
                     "Biology", "Chemistry", "Healthcare Management"],
    "Software Engineer": ["Graphic Design", "Content Marketing", "Financial Analysis",
                         "Biology", "Healthcare", "Psychology"],
    "AI/ML Engineer": ["Accounting", "Sales", "Marketing", "Customer Support",
                      "Project Management", "Content Creation"]
}

# Experience levels
experience_levels = ["Entry-level", "Junior", "Mid-level", "Senior", "Lead"]

# Education backgrounds
education = [
    "Bachelor's in Computer Science", "Master's in Computer Science",
    "Bachelor's in Data Science", "Master's in Data Science",
    "Bachelor's in Engineering", "Master's in Engineering",
    "Bachelor's in Information Technology", "Master's in Information Technology",
    "Bachelor's in Mathematics", "Master's in Mathematics",
    "PhD in Computer Science", "PhD in Statistics"
]

# Generate data
data = []

# For each role, generate multiple entries
for role in roles:
    # Generate positive examples (suitable resumes)
    for i in range(30):  # 30 suitable resumes per role
        # Select random experience level
        exp_level = random.choice(experience_levels)
        edu = random.choice(education)

        # Select core skills (always include at least 3-5 core skills for suitable resumes)
        core_skills = random.sample(role_skills[role][:15], random.randint(3, 5))

        # Select some related skills
        related_skills = random.sample(role_skills[role][15:25], random.randint(2, 4))

        # Select some background skills
        available_background = len(role_skills[role][25:])
        if available_background > 0:
            sample_size = min(random.randint(1, 3), available_background)
            background_skills = random.sample(role_skills[role][25:], sample_size)
        else:
            background_skills = []


        # Create a realistic resume text
        selected_skills = core_skills + related_skills + background_skills
        random.shuffle(selected_skills)

        years_exp = random.randint(1, 10)

        resume_text = f"{exp_level} professional with {years_exp} years of experience. {edu}. "
        resume_text += f"Proficient in {', '.join(selected_skills[:3])}. "
        resume_text += f"Experienced with {', '.join(selected_skills[3:6])}. "

        # Add some project experience
        resume_text += f"Completed projects involving {', '.join(selected_skills[6:8])}. "
        resume_text += f"Strong background in {', '.join(selected_skills[8:])}."

        data.append({
            "Role": role,
            "Text": resume_text,
            "Outcome": 1  # Suitable
        })

    # Generate negative examples (unsuitable resumes)
    for i in range(20):  # 20 unsuitable resumes per role
        exp_level = random.choice(experience_levels)
        edu = random.choice(education)

        # For unsuitable resumes, use fewer core skills and more unrelated skills
        if random.random() < 0.5:
            # Some unsuitable resumes have some relevant skills but not enough
            core_skills = random.sample(role_skills[role][:15], random.randint(0, 2))
            related_skills = random.sample(role_skills[role][15:25], random.randint(0, 2))
            non_relevant = random.sample(non_matching_skills[role], random.randint(3, 5))
        else:
            # Others are completely unrelated
            core_skills = []
            related_skills = []
            non_relevant_pool = non_matching_skills[role]
            sample_size = min(len(non_relevant_pool), random.randint(5, 7))
            non_relevant = random.sample(non_relevant_pool, sample_size)

        selected_skills = core_skills + related_skills + non_relevant
        random.shuffle(selected_skills)

        years_exp = random.randint(1, 10)

        resume_text = f"{exp_level} professional with {years_exp} years of experience. {edu}. "
        resume_text += f"Skills include {', '.join(selected_skills[:3])}. "
        resume_text += f"Knowledge of {', '.join(selected_skills[3:6])}. "

        # Add some project experience
        if len(selected_skills) > 6:
            resume_text += f"Worked on projects involving {', '.join(selected_skills[6:])}. "

        data.append({
            "Role": role,
            "Text": resume_text,
            "Outcome": 0  # Not suitable
        })

# Create a DataFrame and shuffle the data
training_resumes = pd.DataFrame(data)
training_resumes = training_resumes.sample(frac=1).reset_index(drop=True)  # Shuffle

# Save to CSV
training_resumes.to_csv("training_resumes.csv", index=False)

print(f"Generated {len(training_resumes)} resume entries in training_resumes.csv")
print(f"Suitable resumes: {sum(training_resumes['Outcome'])}")
print(f"Unsuitable resumes: {len(training_resumes) - sum(training_resumes['Outcome'])}")
