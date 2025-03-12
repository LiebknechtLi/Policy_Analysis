setwd("E:/中国科技政策")

getwd()

#install.packages("zoo")
install.packages("Rcpp")
install.packages("brms")  
library(Rcpp)
library(brms)
library(readxl)
library(zoo)
library(ggplot2)

# Read the data file
reg_data <- read_excel("贝叶斯回归数据.xlsx")

# Filling NA values
reg_data$'人员全时当量增量（万人年）' <- na.approx(reg_data$'人员全时当量增量（万人年）', rule=2)
reg_data$科技支出占公共支出比例 <- na.approx(reg_data$科技支出占公共支出比例, rule=2)
reg_data$专利授予数量 <- na.approx(reg_data$专利授予数量, rule=2)
reg_data$'R&D机构数' <- na.approx(reg_data$'R&D机构数', rule=2)

# Display structure and dimensions
str(reg_data)
dim(reg_data)

# Create English version of the dataset by copying and renaming
reg_data_eng <- reg_data
colnames(reg_data_eng)[colnames(reg_data_eng) == "年份"] <- "Year"
colnames(reg_data_eng)[colnames(reg_data_eng) == "人员全时当量增量（万人年）"] <- "FTE_Personnel_Increment"
colnames(reg_data_eng)[colnames(reg_data_eng) == "会议主题建模指数"] <- "Conference_Topic_Index"
colnames(reg_data_eng)[colnames(reg_data_eng) == "R&D机构数"] <- "RnD_Institutions"
colnames(reg_data_eng)[colnames(reg_data_eng) == "科技支出占公共支出比例"] <- "Tech_Expenditure_Proportion"
colnames(reg_data_eng)[colnames(reg_data_eng) == "专利授予数量"] <- "Patents_Granted"

# Check the structure of the new dataset
str(reg_data_eng)

# Standardize all numeric variables except Year and Conference_Topic_Index
reg_data_eng$FTE_Personnel_Increment <- scale(reg_data_eng$FTE_Personnel_Increment)
reg_data_eng$RnD_Institutions <- scale(reg_data_eng$RnD_Institutions)
reg_data_eng$Tech_Expenditure_Proportion <- scale(reg_data_eng$Tech_Expenditure_Proportion)
reg_data_eng$Patents_Granted <- scale(reg_data_eng$Patents_Granted)

# Models with standardized variables (except Year and Conference_Topic_Index)
model_1 <- brm(
  FTE_Personnel_Increment ~ Conference_Topic_Index + (1 | Year), 
  data = reg_data_eng, 
  family = gaussian(),  
  prior = c(prior(normal(0, 10), class = "b")),  
  iter = 4000, chains = 4, cores = 4  
)

model_2 <- brm(
  FTE_Personnel_Increment ~ Conference_Topic_Index + RnD_Institutions + (1 | Year), 
  data = reg_data_eng, 
  family = gaussian(),  
  prior = c(prior(normal(0, 10), class = "b")),  
  iter = 4000, chains = 4, cores = 4  
)

model_3 <- brm(
  FTE_Personnel_Increment ~ Conference_Topic_Index + RnD_Institutions + 
    Tech_Expenditure_Proportion + (1 | Year), 
  data = reg_data_eng, 
  family = gaussian(),  
  prior = c(prior(normal(0, 10), class = "b")),  
  iter = 4000, chains = 4, cores = 4  
)

model_4 <- brm(
  FTE_Personnel_Increment ~ Conference_Topic_Index + RnD_Institutions + 
    Tech_Expenditure_Proportion + Patents_Granted + (1 | Year), 
  data = reg_data_eng, 
  family = gaussian(),  
  prior = c(prior(normal(0, 10), class = "b")),  
  iter = 4000, chains = 4, cores = 4  
)

# Notice: model_5 has a duplicate of RnD_Institutions, keeping it as is for consistency
model_5 <- brm(
  FTE_Personnel_Increment ~ Conference_Topic_Index + RnD_Institutions + 
    Tech_Expenditure_Proportion + Patents_Granted + RnD_Institutions + (1 | Year), 
  data = reg_data_eng, 
  family = gaussian(),  
  prior = c(prior(normal(0, 10), class = "b")),  
  iter = 4000, chains = 4, cores = 4  
)

# Generate the main regression table with English variable names
stargazer(model_1, model_2, model_3, model_4, model_5, type = "text",
          title = "Progressive Regression Models Predicting Full-time Equivalent Personnel Increment (Standardized)",
          column.labels = c("Model 1", "Model 2", "Model 3", "Model 4", "Model 5"),
          covariate.labels = c("Conference Topic Index", 
                               "Number of R&D Institutions (std)",
                               "Tech Expenditure Proportion (std)", 
                               "Patents Granted (std)",
                               "Number of R&D Institutions (std)"), # Duplicate variable in Model 5
          out = "Table_English_Standardized.txt",
          notes = "Note: FTE_Personnel_Increment, RnD_Institutions, Tech_Expenditure_Proportion, and Patents_Granted were standardized prior to analysis.")

## Robustness Checks

# 1. Student's t error distribution (robust to outliers)
robust_model_1 <- brm(
  FTE_Personnel_Increment ~ Conference_Topic_Index + RnD_Institutions + 
    Tech_Expenditure_Proportion + Patents_Granted + (1 | Year), 
  data = reg_data_eng, 
  family = student(),  # Student's t distribution for robustness to outliers
  prior = c(prior(normal(0, 10), class = "b"),
            prior(gamma(3, 1), class = "nu")),  # prior for degrees of freedom
  iter = 4000, chains = 4, cores = 4  
)

# 2. Different priors
robust_model_2 <- brm(
  FTE_Personnel_Increment ~ Conference_Topic_Index + RnD_Institutions + 
    Tech_Expenditure_Proportion + Patents_Granted + (1 | Year), 
  data = reg_data_eng, 
  family = gaussian(),  
  prior = c(prior(normal(0, 5), class = "b")),  # More informative prior
  iter = 4000, chains = 4, cores = 4  
)

# 3. Examining interaction effects
robust_model_3 <- brm(
  FTE_Personnel_Increment ~ Conference_Topic_Index * RnD_Institutions + 
    Tech_Expenditure_Proportion + Patents_Granted + (1 | Year), 
  data = reg_data_eng, 
  family = gaussian(),  
  prior = c(prior(normal(0, 10), class = "b")),  
  iter = 4000, chains = 4, cores = 4  
)

# Generate robustness check table
stargazer(model_4, robust_model_1, robust_model_2, robust_model_3, type = "text",
          title = "Robustness Checks for Full-time Equivalent Personnel Increment Models",
          column.labels = c("Original", "Student-t", "Informative Priors", "Interaction"),
          covariate.labels = c("Conference Topic Index", 
                               "Number of R&D Institutions (std)",
                               "Tech Expenditure Proportion (std)", 
                               "Patents Granted (std)",
                               "Conference Topic Index × R&D Institutions"),
          notes = "All models use standardized variables except Year and Conference_Topic_Index. Model 1: Original specification with Gaussian error. Model 2: Student's t error distribution for outlier robustness. Model 3: More informative priors. Model 4: Interaction between Conference Topic Index and Number of R&D Institutions.",
          out = "Robustness_Checks.txt")

# Compare models using LOO (Leave-One-Out cross-validation)
loo_model4 <- loo(model_4)
loo_robust1 <- loo(robust_model_1)
loo_robust2 <- loo(robust_model_2)
loo_robust3 <- loo(robust_model_3)

# Compare models
loo_compare(loo_model4, loo_robust1, loo_robust2, loo_robust3)