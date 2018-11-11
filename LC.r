loan <- read.csv("loan.csv", stringsAsFactors = FALSE)
loanT <- loan

loan$dti <- ifelse(!is.na(loan$dti_joint), loan$dti_joint, loan$dti)
loan$annual_inc <- ifelse(!is.na(loan$annual_inc_joint), loan$annual_inc_joint, loan$annual_inc)
num.NA <- sort(sapply(loan, function(x) {sum(is.na(x))}), decreasing=TRUE)
remain.col <- names(num.NA)[which(num.NA <= 0.8 * dim(loan)[1])]
loan <- loan[, remain.col]
loan$home_ownership <- ifelse(loan$home_ownership %in% c('ANY', 'NONE', 'OTHER'), 'OTHER',
                              loan$home_ownership)
int_state <- by(loan, loan$addr_state, function(x) {
  return(mean(x$int_rate))
})
loan$state_mean_int <-
  ifelse(loan$addr_state %in% names(int_state)[which(int_state <=
                                                       quantile(int_state, 0.25))], 'low',
         ifelse(loan$addr_state %in% names(int_state)[which(int_state <=
                                                              quantile(int_state, 0.5))],'lowmedium',
                ifelse(loan$addr_state %in% names(int_state)[which(int_state <= quantile(int_state, 0.75))], 
                       'mediumhigh', 'high')))
loan$tot_cur_bal[which(is.na(loan$tot_cur_bal))] <- median(loan$tot_cur_bal, na.rm = T)
loan$total_acc[which(is.na(loan$total_acc))] <- median(loan$total_acc, na.rm = T)
loan$open_acc[which(is.na(loan$open_acc))] <- median(loan$open_acc, na.rm = T)
loan$annual_inc[which(is.na(loan$annual_inc))] <- median(loan$annual_inc, na.rm = T)
# Build model and evaluate performance
# split data into train and test for model performance
set.seed(1)
train.ind <- sample(1:dim(loan)[1], 0.7 * dim(loan)[1])
train <- loan[train.ind, ]
test <- loan[-train.ind, ]
train.sub <- train[, c('int_rate', 'state_mean_int', 'home_ownership', 'annual_inc', 'dti',
                       'term', 'loan_amnt', 'total_acc', 'tot_cur_bal', 'open_acc')]

mod2 <- lm(int_rate ~ ., data = train.sub)
summary(mod2)

# See extremely small estimate, e.g., loan_amnt, because of the magnitude, to compare the relative importance of features
# Standardize
train.sub.scale <- train.sub
train.sub.scale[, c(4,5,7,8,9,10)] <- scale(train.sub.scale[, c(4,5,7,8,9,10)])

mod3 <- lm(int_rate ~ ., data = train.sub.scale)
# standardizing won't change the significant of features, but the estimate will change.

# Rows with any NA will be removed.
train.sub.matrix <- model.matrix( ~., train.sub)
head(train.sub.matrix)

x <- train.sub.matrix[, -2]
y <- train.sub.matrix[, 2]
# to calculate the XT*X
t(x) %*% x
# If there is error, due to only taking matrix as argument
# x <- as.matrix(x)

# note that X dim is n * (p+1), XT*X dim is (p+1) * (p+1)
# inverse
xtxi <- solve(t(x) %*% x)
# beta estimator
xtxi %*% t(x) %*% y
# compare with model fitted coefficient
coef(mod2)

# sigma estimator, there are 13 features in total, not including beta0, 
# res = actual - fitted
head(mod2$res)
# MSE:
sqrt(sum(mod2$res^2)/(dim(train.sub)[1] - 14)) # dim(train.sub)[1] - 14 = 621151 degree of freedom

# R square: 1 - sum_square_residual / sum_square_total
1 - sum(mod2$res^2)/sum((y-mean(y))^2)
# adjusted R square
1 - (sum(mod2$res^2)/sum((y-mean(y))^2)) * 
  (dim(train.sub)[1] - 1) /(dim(train.sub)[1] - 13 - 1)
# small p, R square adjusted is very similar to R square.

# F test score, go to slides.
sst = sum((y - mean(y))^2) # sum of square total, df = n - 1 = 621164
ssr = sum(mod2$res^2) #  sum of square residual, df = n-1-p = n - 14= 621151
ssm = sum((y - mean(y))^2) - sum(mod2$res^2) # sum of square model, df = 13
Fstats = (ssm)/(13) / (ssr / (dim(train.sub)[1] - 13 -1))
1 - pf(Fstats, 13, (dim(train.sub)[1] - 13 - 1)) # def = p and n-1-p

# residual = observed - fitted
head(sort(mod2$res))
mod2$res[which.min(mod2$res)]
mod2$res[which.max(mod2$res)]
plot(mod2$fit, mod2$res, xlab = 'Fitted', ylab = 'residual')

# See data points with negative fitted value, we should not predict negative interest rate
mod2_1 <- lm(log(int_rate) ~ state_mean_int + home_ownership + annual_inc + dti +
               + term + loan_amnt + total_acc + tot_cur_bal + open_acc,
             data = train.sub)
# fitted or predicted interest rate
summary(exp(mod2_1$fitted.values))
summary(mod2_1$residuals)
plot(exp(mod2_1$fit), train.sub$int_rate -exp(mod2_1$fit), xlab = 'Fitted', ylab = 'residual')

# still large residuals for some data points. Check the reason.
pred <- round(exp(predict(mod2_1, train.sub)), 2)
ind <- which(mod2_1$fitted <= log(4.5))
cbind(train.sub[ind, ], pred = pred[ind], 
      res = train.sub[ind, 'int_rate'] - pred[ind])

mod2_2 <- lm(log(int_rate) ~ state_mean_int + home_ownership + log(annual_inc) + dti +
               + term + loan_amnt + total_acc + tot_cur_bal + open_acc,
             data = train.sub)
summary(exp(mod2_2$fitted.values))
# plot(mod2)
# first plot we can check unbiased/biased and homo/hetero of the residual
# Def not having homo, reason is model miss important features.
# second plot to check the normality of the residual. 
# qqplot: for ith percentile data point, find ith percentile in normal distribution.

library(glmnet)
# glmnet only takes matrix, can use is.data.frame() or is.matrix() to test
# glmnet standardizes every feature, even categorical feature
# http://stackoverflow.com/questions/17887747/how-does-glmnets-standardize-argument-handle-dummy-variables

ind = train.sub[, -1]
ind <- model.matrix( ~., ind)
dep <- train.sub[, 1]
fit1 <- glmnet(x=ind, y=dep) # default is alpha = 1, lasso
plot(fit1, label = T)

ind = train.sub.scale[, -1]
ind <- model.matrix( ~., ind)
dep <- train.sub.scale[, 1]
fit <- glmnet(x=ind, y=dep) # default is alpha = 1, lasso
plot(fit, label = T)
par(mfrow = c(1, 2))
# Understand the plot
# The top row indicates the number of nonzero coefficients at the current λ,
# which is the effective degrees of freedom (df) for the lasso.
# y axis is the value of coefficient
# x axis is the sum of absolute value of coefficients (L1 norm), or log(lambda)
plot(fit, label = T)
plot(fit, xvar = "lambda", label = T)
vnat=coef(fit) # why do we see two intercepts, one is from model.matrix, one is default added in glmnet
vnat <- vnat[-c(1,2), ncol(vnat)] # remove the intercept, and get the coefficients at the end of the path
# default is par(mar=c(5.1,4.1,4.1,2.1), bottom, lef, top, right
par(mar = c(5.1,6,4.1,2.1))
plot(fit, xvar = 'lambda', label = T, yaxt='n', ylab = "")
axis(2, at=vnat,line=-.5,label = colnames(ind)[-1],las=1, cex.axis=0.5)

print(fit)
# Df is the non zero beta, 
# saturated model is a model with a parameter for every observation so that the data are fitted exactly.
# Deviance_model = 2*(loglikelihood_saturate_model - loglikelihood_current_model)
# Deviance_null = 2*(loglikelihood_saturate_model - loglikelihood_intercept_only_model)
# Deviance percentage = 1 -  Deviance_model / Deviance_null
# lambda value

coef(fit, s = 1/exp(2)) # s stands for lambda
coef(fit, s = 1/exp(8))

# We can choose lambda by checking the picture, Still kinda subjective
# use cross validation to get optimal value of lambda, 
cvfit <- cv.glmnet(ind, dep)
plot(cvfit)
# Two selected lambdas are shown, 
cvfit$lambda.min # value of lambda gives minimal mean cross validated error
cvfit$lambda.1se # most regularized model such that error is within one std err of the minimum
x = coef(cvfit, s = "lambda.min")
coef(cvfit, s = "lambda.1se")

# logisitic regression
sort(table(loan$loan_status))
loan$loan_status <- gsub('Does not meet the credit policy. Status:',
                         '', loan$loan_status)
sort(table(loan$loan_status))
loan <- subset(loan, !loan_status %in% c('Current', 'Issued'))
loan$loan_status_binary <- with(loan, ifelse(loan_status == 'Fully Paid', 1, 0))
loan$log_annual_inc <- log(loan$annual_inc + 1)
train.ind <- sample(1:dim(loan)[1], 0.7 * dim(loan)[1])
train.sub <- loan[train.ind, c('loan_status_binary', 'state_mean_int', 'home_ownership', 'log_annual_inc', 'dti',
                       'term', 'loan_amnt', 'total_acc', 'tot_cur_bal', 'open_acc')]
train.sub$state_mean_int <- relevel(as.factor(train.sub$state_mean_int), ref = 'low')
logis.mod <- glm(loan_status_binary ~ ., train.sub, family = 'binomial')
summary(logis.mod)
test <-  loan[-train.ind, c('loan_status_binary', 'state_mean_int', 'home_ownership', 'log_annual_inc', 'dti',
                           'term', 'loan_amnt', 'total_acc', 'tot_cur_bal', 'open_acc')]
library(pROC)
pred <- predict(logis.mod, test)
plot.roc(test$loan_status_binary, pred)

train.sub.scale <- train.sub
train.sub.scale[, c(4,5,7,8,9,10)] <- scale(train.sub.scale[, c(4,5,7,8,9,10)])

ind = train.sub.scale[, -1]
ind <- model.matrix( ~., ind)
dep <- train.sub.scale[, 1]
logis.cvfit <- cv.glmnet(ind, dep, family = 'binomial')
plot(logis.cvfit)
