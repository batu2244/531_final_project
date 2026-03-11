import os
import random
from datetime import datetime


# --- Configure environment --- 

# Input resume directory
RESUME_DIRECTORY = './Resumes/Tech'

# Output resume directory
now = datetime.now()
time_string = now.strftime('%H:%M:%S')

OUTPUT_DIRECTORY = f'./Output_Resumes/{time_string}'

# Number of resumes to select
SAMPLE_SIZE = 4
NUM_WHITE = SAMPLE_SIZE // 2
NUM_BLACK = SAMPLE_SIZE - NUM_WHITE

# Initialize name banks
WHITE_FIRST = ['Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Kristen', 'Meredith', 'Sarah',
               'Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd']
WHITE_LAST = ['Baker', 'Kelly', 'McCarthy', 'Murphy', 'Murray', "O’Brien", 'Ryan', 'Sullivan', 'Walsh']

BLACK_FIST = ['Aisha', 'Ebony', 'Keisha', 'Latonya', 'Kenya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha',
              'Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone']
BLACK_LAST = ['Jackson', 'Jones', 'Robinson', 'Washington', 'Williams']

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# --- Random sampling ---

# Gather resumes
resume_files = [file for file in os.listdir(RESUME_DIRECTORY)]

# Sample resumes
selected_resumes = random.sample(resume_files, SAMPLE_SIZE)

# Randomly select names
white_first_sample = random.sample(WHITE_FIRST, NUM_WHITE)
white_last_sample = random.sample(WHITE_LAST, NUM_WHITE)
black_first_sample = random.sample(BLACK_FIST, NUM_BLACK)
black_last_sample = random.sample(BLACK_LAST, NUM_BLACK)

# Create full names
white_names = [first + ' ' + last for first, last in zip(white_first_sample, white_last_sample)]
black_names = [first + ' ' + last for first, last in zip(black_first_sample, black_last_sample)]

# Create list of names
names = white_names + black_names
random.shuffle(names)

# Add names to resumes
for i, resume in enumerate(selected_resumes):
    # Copy resume text
    with open(f'{RESUME_DIRECTORY}/{resume}', 'r') as f:
        base_content = f.read()

    # Write new resume
    with open(f'{OUTPUT_DIRECTORY}/{names[i]}', 'w') as f:
        f.write(f'Name: {names[i]}' + '\n\n' + base_content)

    print('Added name:', names[i])