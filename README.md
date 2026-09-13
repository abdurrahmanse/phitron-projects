# Phitron Projects

Jupyter Notebook assignments, midterm & final exam solutions, and datasets from [Phitron](https://phitron.io) coursework.

## Courses

| Course | Folder | Contents |
|---|---|---|
| 🐍 AI Programming With Python | `AI Programming With Python/` | 3 assignments, midterm, final exam |
| 🤖 Machine Learning | `Machine Learning/` | 3 assignments, final exam |
| 🧠 Deep Learning | `Deep learning/` | Midterm, final exam |

## Structure

```
├── AI Programming With Python/
│   ├── Assignment/          # Assignment_01, midterm_solutions, Final_Exam_Question
│   ├── Assignment-2/        # Assignment_01, Assignment_02, app
│   └── Assignment-3/        # index
├── Machine Learning/
│   ├── Assignment/          # ML_Assignment_01–03, ML_FINAL
│   └── data/                # diabetes.csv
├── Deep learning/
│   ├── Final/               # DL Final Exam
│   └── mid/                 # Midterm Exam
└── data/                    # Shared datasets + 50+ beginner CSVs
```

## Datasets

`data/` includes `titanic`, `adult_income`, `house_price_regression`, `masters_data_science_performance`, and a `beginner_datasets/` folder with 50+ CSVs (iris, diabetes, boston, heart, amazon, gold, and more).

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

> Deep Learning notebooks may require `tensorflow` or `torch`.

## Usage

```bash
git clone https://github.com/abdurrahmanse/phitron-projects.git
cd phitron-projects
jupyter notebook
```

Keep the `data/` folders in place — notebooks rely on relative paths.

---

**Author:** Abdur Rahman · [@abdurrahmanse](https://github.com/abdurrahmanse)

