# train.R

# 加载所需库
library(keras)
library(ggplot2)

# 加载自定义模块
source("CNNAVG.R")

# 加载MNIST数据
mnist <- dataset_mnist()
x_train <- mnist$train$x / 255
y_train <- to_categorical(mnist$train$y, 10)
x_train <- array_reshape(x_train, c(dim(x_train)[1], 28, 28, 1))

x_test <- mnist$test$x / 255
y_test <- to_categorical(mnist$test$y, 10)
x_test <- array_reshape(x_test, c(dim(x_test)[1], 28, 28, 1))

# 模拟非IID数据分布
num_clients <- 10
client_data <- lapply(1:num_clients, function(i) {
  indices <- sample(1:nrow(x_train), size = nrow(x_train) / num_clients)
  list(x = x_train[indices, , , , drop = FALSE], y = y_train[indices, ])
})

# 初始化全局模型和最终评估模型
global_model_weights <- initialize_global_model()
final_model <- create_final_model()

# 联邦学习主循环
num_rounds <- 10
history <- data.frame(round = numeric(), accuracy = numeric(), loss = numeric())

for (round in 1:num_rounds) {
  cat("Round:", round, "\n")
  
  # 随机选择部分客户端
  selected_clients <- sample(1:num_clients, size = 5)
  
  # 客户端训练并收集更新
  client_updates <- lapply(selected_clients, function(client_id) {
    client_train(client_data[[client_id]], global_model_weights)
  })
  
  # 获取客户端数据量
  client_sizes <- sapply(selected_clients, function(client_id) {
    nrow(client_data[[client_id]]$x)
  })
  
  # 聚合权重
  global_model_weights <- aggregate_weights(client_updates, client_sizes)
  
  # 设置全局模型权重并评估
  final_model$set_weights(global_model_weights)
  metrics <- final_model %>% evaluate(x_test, y_test, verbose = 0)
  
  cat("Test Accuracy:", metrics["accuracy"], "\n")
  history <- rbind(history, data.frame(round = round, accuracy = metrics["accuracy"], loss = metrics["loss"]))
}

# 可视化准确率变化
ggplot(history, aes(x = round)) +
  geom_line(aes(y = accuracy), color = "blue") +
  ggtitle("Model Accuracy over Rounds") +
  xlab("Round") + ylab("Accuracy") +
  theme_minimal()

# 可视化损失变化
ggplot(history, aes(x = round)) +
  geom_line(aes(y = loss), color = "red") +
  ggtitle("Model Loss over Rounds") +
  xlab("Round") + ylab("Loss") +
  theme_minimal()
