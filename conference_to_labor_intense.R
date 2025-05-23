setwd("E:/中国科技政策")

getwd()

# 加载必要的包
library(readxl)
library(zoo)
library(lme4)
library(lmtest)
library(sandwich)  
library(car)       
library(MASS)      
library(tseries)   


# 读取数据文件
reg_data <- read_excel("贝叶斯回归数据.xlsx")

# 填充缺失值
reg_data$'人员全时当量增量（万人年）' <- na.approx(reg_data$'人员全时当量增量（万人年）', rule=2)
reg_data$科技支出占公共支出比例 <- na.approx(reg_data$科技支出占公共支出比例, rule=2)
reg_data$专利授予数量 <- na.approx(reg_data$专利授予数量, rule=2)
reg_data$'R&D机构数' <- na.approx(reg_data$'R&D机构数', rule=2)

# 创建英文版本的数据集
reg_data_eng <- reg_data
colnames(reg_data_eng)[colnames(reg_data_eng) == "年份"] <- "Year"
colnames(reg_data_eng)[colnames(reg_data_eng) == "人员全时当量增量（万人年）"] <- "FTE_Personnel_Increment"
colnames(reg_data_eng)[colnames(reg_data_eng) == "会议主题建模指数"] <- "Conference_Topic_Index"
colnames(reg_data_eng)[colnames(reg_data_eng) == "R&D机构数"] <- "RnD_Institutions"
colnames(reg_data_eng)[colnames(reg_data_eng) == "科技支出占公共支出比例"] <- "Tech_Expenditure_Proportion"
colnames(reg_data_eng)[colnames(reg_data_eng) == "专利授予数量"] <- "Patents_Granted"

# 确保数据类型正确
reg_data_eng$Year <- as.numeric(reg_data_eng$Year)
reg_data_eng$FTE_Personnel_Increment <- as.numeric(reg_data_eng$FTE_Personnel_Increment)
reg_data_eng$Conference_Topic_Index <- as.numeric(reg_data_eng$Conference_Topic_Index)
reg_data_eng$RnD_Institutions <- as.numeric(reg_data_eng$RnD_Institutions)
reg_data_eng$Tech_Expenditure_Proportion <- as.numeric(reg_data_eng$Tech_Expenditure_Proportion)
reg_data_eng$Patents_Granted <- as.numeric(reg_data_eng$Patents_Granted)

# 按年份排序数据
reg_data_eng <- reg_data_eng[order(reg_data_eng$Year), ]

# 输出排序后的数据
print("数据排序后的年份顺序:")
print(reg_data_eng$Year)

# 创建滞后变量
# 将会议主题建模指数放大10000倍
reg_data_eng$Conference_Topic_Index <- reg_data_eng$Conference_Topic_Index * 10000

# 为独立变量创建滞后1年的版本
reg_data_eng$Conference_Topic_Index_Lag1 <- c(NA, head(reg_data_eng$Conference_Topic_Index, -1))
reg_data_eng$RnD_Institutions_Lag1 <- c(NA, head(reg_data_eng$RnD_Institutions, -1))
reg_data_eng$Tech_Expenditure_Proportion_Lag1 <- c(NA, head(reg_data_eng$Tech_Expenditure_Proportion, -1))
reg_data_eng$Patents_Granted_Lag1 <- c(NA, head(reg_data_eng$Patents_Granted, -1))

# 输出创建的滞后变量的前几行
print("原始和滞后变量对比:")
print(head(reg_data_eng[, c("Year", "Conference_Topic_Index", "Conference_Topic_Index_Lag1")]))

# 删除第一行（因为它的滞后变量是NA）
reg_data_lag <- reg_data_eng[-1, ]

# 检查数据中的NA值
print("最终数据集中的NA值数量:")
print(colSums(is.na(reg_data_lag)))

# 标准化控制变量，但保留因变量和会议主题指数的原始值
reg_data_lag$RnD_Institutions_Lag1 <- as.numeric(scale(reg_data_lag$RnD_Institutions_Lag1))
reg_data_lag$Tech_Expenditure_Proportion_Lag1 <- as.numeric(scale(reg_data_lag$Tech_Expenditure_Proportion_Lag1))
reg_data_lag$Patents_Granted_Lag1 <- as.numeric(scale(reg_data_lag$Patents_Granted_Lag1))

# 转换Year为因子类型用于固定效应
reg_data_lag$Year_Factor <- as.factor(reg_data_lag$Year)

# 简化构建模型 - 不使用过多的进阶模型以避免错误
model_lag_1 <- lm(FTE_Personnel_Increment ~ Conference_Topic_Index_Lag1, data = reg_data_lag)
model_lag_2 <- lm(FTE_Personnel_Increment ~ Conference_Topic_Index_Lag1 + RnD_Institutions_Lag1, data = reg_data_lag)
model_lag_3 <- lm(FTE_Personnel_Increment ~ Conference_Topic_Index_Lag1 + RnD_Institutions_Lag1 + 
                    Tech_Expenditure_Proportion_Lag1, data = reg_data_lag)
model_lag_4 <- lm(FTE_Personnel_Increment ~ Conference_Topic_Index_Lag1 + RnD_Institutions_Lag1 + 
                    Tech_Expenditure_Proportion_Lag1 + Patents_Granted_Lag1, data = reg_data_lag)

# 创建交互项模型
model_lag_interaction <- lm(FTE_Personnel_Increment ~ Conference_Topic_Index_Lag1 * RnD_Institutions_Lag1 + 
                              Tech_Expenditure_Proportion_Lag1 + Patents_Granted_Lag1, data = reg_data_lag)

# 创建带年份固定效应的模型
model_lag_year_fe <- lm(FTE_Personnel_Increment ~ Conference_Topic_Index_Lag1 + RnD_Institutions_Lag1 + 
                          Tech_Expenditure_Proportion_Lag1 + Patents_Granted_Lag1 + Year_Factor, data = reg_data_lag)


# 创建系数汇总表格
results_table <- data.frame(
  Variable = c("Intercept", "Conference_Topic_Index_Lag1", "RnD_Institutions_Lag1", 
               "Tech_Expenditure_Proportion_Lag1", "Patents_Granted_Lag1"),
  Model1 = c(coef(model_lag_1)[1], coef(model_lag_1)[2], NA, NA, NA),
  Model2 = c(coef(model_lag_2)[1], coef(model_lag_2)[2], coef(model_lag_2)[3], NA, NA),
  Model3 = c(coef(model_lag_3)[1], coef(model_lag_3)[2], coef(model_lag_3)[3], coef(model_lag_3)[4], NA),
  Model4 = c(coef(model_lag_4)[1], coef(model_lag_4)[2], coef(model_lag_4)[3], coef(model_lag_4)[4], coef(model_lag_4)[5])
)

models <- list(
  "M1"= model_lag_1,
  "M2"= model_lag_2,
  "M3"= model_lag_3,
  "M4"= model_lag_4,
  "M5"= model_lag_interaction,
  "M6"= model_lag_year_fe
)

modelsummary(models, 
             stars = TRUE, 
             statistic = "std.error",  # Displays robust SEs
             output = "regression_of_labor.html")


# 1. 使用稳健标准误
cat("\n1. 使用稳健标准误 (HC3) 进行模型4的检验:\n")
robust_se <- coeftest(model_lag_4, vcov = vcovHC(model_lag_4, type = "HC3"))
print(robust_se)

# 2. 进行稳健回归 (使用M估计，对异常值不敏感)
cat("\n2. 使用稳健回归 (M估计):\n")
robust_model <- rlm(FTE_Personnel_Increment ~ Conference_Topic_Index_Lag1 + RnD_Institutions_Lag1 + 
                      Tech_Expenditure_Proportion_Lag1 + Patents_Granted_Lag1, 
                    data = reg_data_lag, method = "M")

robust_model2 <-  rlm(FTE_Personnel_Increment ~ Conference_Topic_Index_Lag1 * RnD_Institutions_Lag1 + 
                        Tech_Expenditure_Proportion_Lag1 + Patents_Granted_Lag1 , 
                      data = reg_data_lag, method = "M")




models2 <- list(
  "M1"= model_lag_1,
  "M2"= model_lag_2,
  "M3"= model_lag_3,
  "M4"= model_lag_4,
  "M5"= model_lag_interaction,
  "M6"= robust_model,
  "M7"= robust_model2
)


modelsummary(models2,
             stars = TRUE, 
             statistic = "std.error",  # Displays robust SEs
             output = "Robust Regression Models.html"
  
  
)

plot(model_lag_4, which=1)

hist(reg_data_eng$Tech_Expenditure_Proportion, main = "Distribution of Tech Expenditure", xlab = "Tech Expenditure")


vif_result <- vif(model_lag_4)
print(vif_result)


library(lmtest)
dw_test <- dwtest(model_lag_4)
print(dw_test)

bp_test <- bptest(model_lag_4)
print(bp_test)

hist(residuals(model_lag_4), breaks = 5, main = "Histogram of Residuals", xlab = "Residuals")


