# 加载所需库
library(keras)
library(ggplot2)

# 初始化全局模型
initialize_global_model <- function() {
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(5, 5), activation = 'relu', input_shape = c(28, 28, 1)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(5, 5), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = 'relu') %>%
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
  
  if (length(client_weights) != length(client_sizes)) {
    stop("client_weights and client_sizes must have the same length.")
  }
  
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
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(5, 5), activation = 'relu', input_shape = c(28, 28, 1)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(5, 5), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
  )
  model$set_weights(global_weights)
  
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
x_train <- array_reshape(x_train, c(dim(x_train)[1], 28, 28, 1))

x_test <- mnist$test$x / 255
y_test <- to_categorical(mnist$test$y, 10)
x_test <- array_reshape(x_test, c(dim(x_test)[1], 28, 28, 1))
print(dim(x_train))

# 模拟非IID数据分布
num_clients <- 10
client_data <- lapply(1:num_clients, function(i) {
  indices <- sample(1:nrow(x_train), size = nrow(x_train) / num_clients)
  list(x = x_train[indices, , , , drop = FALSE], y = y_train[indices, ])
})

final_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(5, 5), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(5, 5), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
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
history <- data.frame(round = numeric(), accuracy = numeric(), loss = numeric())

for (round in 1:num_rounds) {
  cat("Round:", round, "\n")
  selected_clients <- sample(1:num_clients, size = 5)
  
  client_updates <- lapply(selected_clients, function(client_id) {
    client_train(client_data[[client_id]], global_model_weights)
  })
  
  client_sizes <- sapply(selected_clients, function(client_id) {
    nrow(client_data[[client_id]]$x)
  })
  global_model_weights <- aggregate_weights(client_updates, client_sizes)
  
  final_model$set_weights(global_model_weights)
  metrics <- final_model %>% evaluate(x_test, y_test, verbose = 0)
  
  cat("Test Accuracy:", metrics["accuracy"], "\n")
  history <- rbind(history, data.frame(round = round, accuracy = metrics["accuracy"], loss = metrics["loss"]))
}

# 可视化
ggplot(history, aes(x = round)) +
  geom_line(aes(y = accuracy), color = "blue") +
  ggtitle("Model Accuracy over Rounds") +
  xlab("Round") + ylab("Accuracy") +
  theme_minimal()

ggplot(history, aes(x = round)) +
  geom_line(aes(y = loss), color = "red") +
  ggtitle("Model Loss over Rounds") +
  xlab("Round") + ylab("Loss") +
  theme_minimal()

