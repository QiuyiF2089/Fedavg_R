# MLP.R

# 初始化全局模型，返回初始权重
initialize_global_model <- function() {
  model <- keras_model_sequential() %>%
    layer_flatten(input_shape = c(28, 28, 1)) %>%
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
    layer_flatten(input_shape = c(28, 28, 1)) %>%
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dense(units = 200, activation = 'relu') %>%
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

# 创建最终模型实例
create_final_model <- function() {
  model <- keras_model_sequential() %>%
    layer_flatten(input_shape = c(28, 28, 1)) %>%
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')
  
  model %>% compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
  )
  
  return(model)
}
