setwd("E:/中国科技政策")

getwd()

# 加载必要的包
library(readxl)
library(zoo)
library(lme4)
library(lmtest)
library(readxl)

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

# 将会议主题建模指数放大10000倍
reg_data_eng$Conference_Topic_Index <- reg_data_eng$Conference_Topic_Index * 10000

# 创建滞后变量 - 为独立变量创建滞后1年的版本
reg_data_eng$Conference_Topic_Index_Lag1 <- c(NA, head(reg_data_eng$Conference_Topic_Index, -1))
reg_data_eng$FTE_Personnel_Increment_Lag1 <- c(NA, head(reg_data_eng$FTE_Personnel_Increment, -1))
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
reg_data_lag$FTE_Personnel_Increment_Lag1 <- as.numeric(scale(reg_data_lag$FTE_Personnel_Increment_Lag1))
reg_data_lag$Tech_Expenditure_Proportion_Lag1 <- as.numeric(scale(reg_data_lag$Tech_Expenditure_Proportion_Lag1))
reg_data_lag$Patents_Granted_Lag1 <- as.numeric(scale(reg_data_lag$Patents_Granted_Lag1))

# 转换Year为因子类型用于固定效应
reg_data_lag$Year_Factor <- as.factor(reg_data_lag$Year)

# 模型构建 - 现在RnD_Institutions为因变量
# 模型1：只有前一年的会议主题建模指数
model_rnd_1 <- lm(RnD_Institutions ~ Conference_Topic_Index_Lag1, data = reg_data_lag)

# 模型2：添加前一年的人员全时当量增量
model_rnd_2 <- lm(RnD_Institutions ~ Conference_Topic_Index_Lag1 + FTE_Personnel_Increment_Lag1, data = reg_data_lag)

# 模型3：添加前一年的科技支出占比
model_rnd_3 <- lm(RnD_Institutions ~ Conference_Topic_Index_Lag1 + FTE_Personnel_Increment_Lag1 + 
                    Tech_Expenditure_Proportion_Lag1, data = reg_data_lag)

# 模型4：添加前一年的专利授予数量
model_rnd_4 <- lm(RnD_Institutions ~ Conference_Topic_Index_Lag1 + FTE_Personnel_Increment_Lag1 + 
                    Tech_Expenditure_Proportion_Lag1 + Patents_Granted_Lag1, data = reg_data_lag)


# 创建交互项模型
model_rnd_interaction <- lm(RnD_Institutions ~ Conference_Topic_Index_Lag1*Tech_Expenditure_Proportion_Lag1 + 
                              FTE_Personnel_Increment_Lag1 + Patents_Granted_Lag1, data = reg_data_lag)

# 创建带年份固定效应的模型 
model_rnd_year_fe <- lm(RnD_Institutions ~ Conference_Topic_Index_Lag1 + FTE_Personnel_Increment_Lag1 + 
                          Tech_Expenditure_Proportion_Lag1 + Patents_Granted_Lag1 + Year_Factor, data = reg_data_lag)


# 创建系数汇总表格
results_table <- data.frame(
  Variable = c("Intercept", "Conference_Topic_Index_Lag1", "FTE_Personnel_Increment_Lag1", 
               "Tech_Expenditure_Proportion_Lag1", "Patents_Granted_Lag1"),
  Model1 = c(coef(model_rnd_1)[1], coef(model_rnd_1)[2], NA, NA, NA),
  Model2 = c(coef(model_rnd_2)[1], coef(model_rnd_2)[2], coef(model_rnd_2)[3], NA, NA),
  Model3 = c(coef(model_rnd_3)[1], coef(model_rnd_3)[2], coef(model_rnd_3)[3], coef(model_rnd_3)[4], NA),
  Model4 = c(coef(model_rnd_4)[1], coef(model_rnd_4)[2], coef(model_rnd_4)[3], coef(model_rnd_4)[4], coef(model_rnd_4)[5])
)


models <- list(
  "M1"= model_rnd_1,
  "M2"= model_rnd_2,
  "M3"= model_rnd_3,
  "M4"= model_rnd_4,
  "M5"= model_rnd_interaction,
  "M6"= model_rnd_year_fe
)

modelsummary(models, 
             stars = TRUE, 
             statistic = "std.error",  # Displays robust SEs
             output = "regression_of_conference.html")


# 进行稳健回归 (使用M估计，对异常值不敏感)
cat("使用稳健回归 (M估计):\n")
robust_model <- rlm(RnD_Institutions ~ Conference_Topic_Index_Lag1 +  FTE_Personnel_Increment +
                      Tech_Expenditure_Proportion_Lag1 + Patents_Granted_Lag1, 
                    data = reg_data_lag, method = "M")
print(summary(robust_model))

plot(model_rnd_4, which=1)

hist(reg_data_eng$RnD_Institutions, main = "Distribution of RnD Institutions", xlab = "RnD Institutions")


vif_result <- vif(model_rnd_4)
print(vif_result)


library(lmtest)
dw_test <- dwtest(model_rnd_4)
print(dw_test)

bp_test <- bptest(model_rnd_4)
print(bp_test)

hist(residuals(model_rnd_4), breaks = 20, main = "Histogram of Residuals", xlab = "Residuals")




