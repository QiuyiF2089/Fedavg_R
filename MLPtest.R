# 加载所需库
library(keras)
library(ggplot2)

# 初始化全局模型
initialize_global_model <- function() {
  model <- keras_model_sequential() %>%
    layer_flatten(input_shape = c(28, 28, 1)) %>%  # 展平输入
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
  )
  
  return(model$get_weights())
}

# 聚合客户端权重 (FedAvg)
aggregate_weights <- function(client_weights, client_sizes) {
  total_samples <- sum(client_sizes)
  
  # 确保权重和样本大小匹配
  if (length(client_weights) != length(client_sizes)) {
    stop("client_weights and client_sizes must have the same length.")
  }
  
  # 加权求和每个客户端的权重
  aggregated_weights <- Reduce(
    function(w1, w2) {
      mapply("+", w1, w2, SIMPLIFY = FALSE)
    },
    lapply(seq_along(client_weights), function(i) {
      lapply(client_weights[[i]], function(w) w * (client_sizes[i] / total_samples))
    })
  )
  
  return(aggregated_weights)
}

# 客户端训练
client_train <- function(data, global_weights, epochs = 1, batch_size = 32) {
  # 重建模型并加载全局权重
  model <- keras_model_sequential() %>%
    layer_flatten(input_shape = c(28, 28, 1)) %>%  # 展平输入
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
  )
  model$set_weights(global_weights)
  
  # 本地训练
  model %>% fit(
    data$x, data$y,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 0
  )
  
  return(model$get_weights())
}

# 加载MNIST数据
mnist <- dataset_mnist()
x_train <- mnist$train$x / 255
y_train <- to_categorical(mnist$train$y, 10)

# 转换输入数据形状为四维 (样本数, 28, 28, 1)
x_train <- array_reshape(x_train, c(dim(x_train)[1], 28, 28, 1))

# 模拟非IID数据分布
num_clients <- 10
client_data <- lapply(1:num_clients, function(i) {
  indices <- sample(1:nrow(x_train), size = nrow(x_train) / num_clients)
  list(x = x_train[indices, , , , drop = FALSE], y = y_train[indices, ])
})

# 加载测试数据
x_test <- mnist$test$x / 255
y_test <- to_categorical(mnist$test$y, 10)

# 转换测试数据形状为四维 (样本数, 28, 28, 1)
x_test <- array_reshape(x_test, c(dim(x_test)[1], 28, 28, 1))

# 重建全局模型并加载最终权重
final_model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(28, 28, 1)) %>%  # 展平输入
  layer_dense(units = 200, activation = 'relu') %>%
  layer_dense(units = 200, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

final_model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# 初始化全局模型
global_model_weights <- initialize_global_model()

# 模拟联邦学习
num_rounds <- 10
history <- data.frame(round = numeric(), accuracy = numeric(), loss = numeric())  # 存储历史性能

for (round in 1:num_rounds) {
  cat("Round:", round, "\n")
  
  # 随机选择客户端
  selected_clients <- sample(1:num_clients, size = 5)
  
  # 收集客户端更新
  client_updates <- lapply(selected_clients, function(client_id) {
    client_train(client_data[[client_id]], global_model_weights)
  })
  
  # 聚合权重
  client_sizes <- sapply(selected_clients, function(client_id) {
    nrow(client_data[[client_id]]$x)
  })
  global_model_weights <- aggregate_weights(client_updates, client_sizes)
  
  # 更新全局模型权重
  final_model$set_weights(global_model_weights)
  
  # 评估模型性能
  metrics <- final_model %>% evaluate(x_test, y_test, verbose = 0)
  cat("Test Accuracy:", metrics["accuracy"], "\n")
  
  # 保存历史性能
  history <- rbind(history, data.frame(round = round, accuracy = metrics["accuracy"], loss = metrics["loss"]))
}

# 可视化训练过程
ggplot(history, aes(x = round)) +
  geom_line(aes(y = accuracy), color = "blue", size = 1) +
  geom_point(aes(y = accuracy), color = "blue") +
  ggtitle("Model Accuracy over Rounds") +
  xlab("Round") +
  ylab("Accuracy") +
  theme_minimal()

ggplot(history, aes(x = round)) +
  geom_line(aes(y = loss), color = "red", size = 1) +
  geom_point(aes(y = loss), color = "red") +
  ggtitle("Model Loss over Rounds") +
  xlab("Round") +
  ylab("Loss") +
  theme_minimal()



