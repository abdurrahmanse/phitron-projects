# Phitron Projects

Jupyter Notebook assignments, midterm & final exam solutions, and datasets from [Phitron](https://phitron.io) coursework.

## Courses

| Course | Folder | Contents |
|---|---|---|
| 🐍 AI Programming With Python | `01_AI_Programming_with_Python/` | 3 assignments, midterm, final exam |
| 🤖 Machine Learning | `02_Machine_Learning/` | 3 assignments, final exam (solution & template) |
| 🧠 Deep Learning | `03_Deep_Learning/` | Midterm (FashionMNIST), final exam (KMNIST) |
| ⚡ Advanced Deep Learning | `04_Advanced_Deep_Learning/` | Midterm (CNNs & face dataset), final exam (RNNs/LSTMs) |

## Structure

```
├── 01_AI_Programming_with_Python/
│   ├── assignments/
│   │   ├── assignment_01/   # Assignment_01.ipynb (Python basics & data structures)
│   │   ├── assignment_02/   # Assignment_01, Assignment_02, app.ipynb
│   │   └── assignment_03/   # index.ipynb
│   ├── midterm/             # Midterm_Solutions.ipynb
│   └── final_exam/          # Final_Exam.ipynb
├── 02_Machine_Learning/
│   ├── assignments/         # ML_Assignment_01–03 (Preprocessing, Regression, Classification)
│   ├── data/                # diabetes.csv
│   └── final_exam/          # ML_Final_Exam.ipynb (Solution), ML_Final_Exam_Question.ipynb
├── 03_Deep_Learning/
│   ├── midterm/             # Midterm_Exam.ipynb & FashionMNIST dataset
│   └── final_exam/          # DL_Final_Exam.ipynb & KMNIST dataset
├── 04_Advanced_Deep_Learning/
│   ├── midterm/             # Advance_DL_Mid_Term_Exam.ipynb, face dataset, best_model.pth, plots
│   └── final_exam/          # Advance_Deep_Learning_Final_Exam.ipynb (RNNs, LSTMs, GRUs)
└── data/                    # Shared datasets + 50+ beginner CSVs
```

## Datasets

- `data/`: Includes `titanic_data_updated`, `adult_income`, `house_price_regression_dataset`, `masters_data_science_performance`, and a `beginner_datasets/` directory with 50+ CSVs (iris, diabetes, boston, heart, amazon, gold, and more).
- `02_Machine_Learning/`: Contains `diabetes.csv` for ML final exam.
- `03_Deep_Learning/`: Contains `FashionMNIST` in `midterm/data/` and `KMNIST` in `final_exam/data/`.
- `04_Advanced_Deep_Learning/midterm/faces/`: Contains facial image training data and labels for age-group classification.

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

> Deep Learning and Advanced Deep Learning notebooks require `torch`, `torchvision`, and/or `tensorflow`.

## Usage

```bash
git clone https://github.com/abdurrahmanse/phitron-projects.git
cd phitron-projects
jupyter notebook
```

Keep dataset folders in their respective locations — notebooks rely on relative paths.

---

**Author:** Abdur Rahman · [@abdurrahmanse](https://github.com/abdurrahmanse)

