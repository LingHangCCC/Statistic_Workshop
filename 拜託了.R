library(dplyr)
library(randomForest)
library(caret)
library(ggplot2)
library(gridExtra)

# ========= 小工具函數 =========

# 在「給定資料」上建立 k 折分層（fold i = 驗證集；訓練集=其餘）
make_stratified_folds <- function(y, k = 5, seed = 123){
  set.seed(seed)
  y <- factor(y)
  folds <- vector("list", k)
  for (lvl in levels(y)){
    idx <- which(y == lvl)
    if (length(idx) < k) {
      stop(sprintf("類別 '%s' 在資料中不足 %d 筆，無法做 %d 折分層 CV。", lvl, k, k))
    }
    idx <- sample(idx)
    fold_sizes <- rep(floor(length(idx)/k), k)
    rem <- length(idx) - sum(fold_sizes)
    if (rem > 0) fold_sizes[seq_len(rem)] <- fold_sizes[seq_len(rem)] + 1
    start <- 1
    for (j in seq_len(k)){
      end <- start + fold_sizes[j] - 1
      folds[[j]] <- c(folds[[j]], idx[start:end])
      start <- end + 1
    }
  }
  lapply(folds, sort)
}

# 安全的 Random Forest 訓練函數
safe_rf_train <- function(formula, data, ntree, mtry = NULL, nodesize = NULL, maxnodes = NULL) {
  tryCatch({
    # 設定預設參數
    if (is.null(mtry)) {
      n_features <- length(all.vars(formula)[-1])
      mtry <- max(1, floor(sqrt(n_features)))
    }
    if (is.null(nodesize)) nodesize <- 5
    
    rf_params <- list(
      formula = formula,
      data = data,
      ntree = ntree,
      mtry = mtry,
      nodesize = nodesize
    )
    
    if (!is.null(maxnodes)) {
      rf_params$maxnodes <- maxnodes
    }
    
    result <- suppressWarnings({
      do.call(randomForest, rf_params)
    })
    
    return(list(model = result, success = TRUE, error = NULL))
  }, error = function(e) {
    return(list(model = NULL, success = FALSE, error = as.character(e)))
  })
}

# ========= 多類別評估指標計算函數 =========

calculate_multiclass_metrics <- function(y_true, y_pred) {
  # 確保都是 factor 且有相同的 levels
  all_levels <- sort(union(levels(y_true), levels(y_pred)))
  y_true <- factor(y_true, levels = all_levels)
  y_pred <- factor(y_pred, levels = all_levels)
  
  # 混淆矩陣
  cm <- confusionMatrix(y_pred, y_true)
  
  # 手動計算每個類別的指標
  n_classes <- length(all_levels)
  class_metrics <- data.frame(
    Class = all_levels,
    Precision = numeric(n_classes),
    Recall = numeric(n_classes),
    F1 = numeric(n_classes),
    Balanced_Accuracy = numeric(n_classes),
    stringsAsFactors = FALSE
  )
  
  # 轉換為數值矩陣便於計算
  cm_matrix <- as.matrix(cm$table)
  
  for (i in seq_along(all_levels)) {
    class_name <- all_levels[i]
    
    # 計算 TP, FP, FN, TN
    tp <- cm_matrix[class_name, class_name]  # True Positive
    fp <- sum(cm_matrix[class_name, ]) - tp  # False Positive
    fn <- sum(cm_matrix[, class_name]) - tp  # False Negative
    tn <- sum(cm_matrix) - tp - fp - fn      # True Negative
    
    # 計算指標
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1 <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
    
    # Balanced Accuracy = (Sensitivity + Specificity) / 2
    sensitivity <- recall  # same as recall
    specificity <- ifelse(tn + fp == 0, 0, tn / (tn + fp))
    balanced_acc <- (sensitivity + specificity) / 2
    
    class_metrics[i, "Precision"] <- precision
    class_metrics[i, "Recall"] <- recall
    class_metrics[i, "F1"] <- f1
    class_metrics[i, "Balanced_Accuracy"] <- balanced_acc
  }
  
  # 計算宏平均和加權平均
  class_support <- table(y_true)
  total_samples <- sum(class_support)
  
  # Macro averages (簡單平均)
  macro_precision <- mean(class_metrics$Precision, na.rm = TRUE)
  macro_recall <- mean(class_metrics$Recall, na.rm = TRUE)
  macro_f1 <- mean(class_metrics$F1, na.rm = TRUE)
  macro_balanced_acc <- mean(class_metrics$Balanced_Accuracy, na.rm = TRUE)
  
  # Weighted averages (依樣本數加權)
  weights <- as.numeric(class_support[class_metrics$Class])
  weighted_precision <- sum(class_metrics$Precision * weights, na.rm = TRUE) / total_samples
  weighted_recall <- sum(class_metrics$Recall * weights, na.rm = TRUE) / total_samples
  weighted_f1 <- sum(class_metrics$F1 * weights, na.rm = TRUE) / total_samples
  weighted_balanced_acc <- sum(class_metrics$Balanced_Accuracy * weights, na.rm = TRUE) / total_samples
  
  # 整體準確度
  overall_accuracy <- sum(diag(cm_matrix)) / sum(cm_matrix)
  
  return(list(
    class_metrics = class_metrics,
    macro_precision = macro_precision,
    macro_recall = macro_recall,
    macro_f1 = macro_f1,
    macro_balanced_accuracy = macro_balanced_acc,
    weighted_precision = weighted_precision,
    weighted_recall = weighted_recall,
    weighted_f1 = weighted_f1,
    weighted_balanced_accuracy = weighted_balanced_acc,
    overall_accuracy = overall_accuracy,
    confusion_matrix = cm
  ))
}

# ========= 主要分析函數 =========

run_rf_separate_files_analysis <- function(train_path = "C:/Users/yu/OneDrive/桌面/train_final_XBG_20.csv",
                                           test_path = "C:/Users/yu/OneDrive/桌面/test_final_XGB_20.csv",
                                           target_col = "Label",
                                           # Random Forest 參數
                                           rf_ntree = 500,
                                           rf_mtry = NULL,
                                           rf_nodesize = 10,
                                           rf_maxnodes = NULL,
                                           # CV 設定
                                           cv_folds = 5,
                                           # 其他設定
                                           save_results = TRUE,
                                           output_dir = "rf_separate_files_results") {
  
  cat("=== 基於已篩選特徵的 5折分層 CV Random Forest 分析 ===\n")
  cat("訓練檔案:", train_path, "\n")
  cat("測試檔案:", test_path, "\n")
  
  analysis_start_time <- Sys.time()
  
  # 1. 讀取訓練和測試資料
  cat("\n步驟 1: 讀取資料...\n")
  
  if (!file.exists(train_path)) {
    stop("找不到訓練資料檔案: ", train_path)
  }
  if (!file.exists(test_path)) {
    stop("找不到測試資料檔案: ", test_path)
  }
  
  train_data <- read.csv(train_path, check.names = TRUE)
  test_data <- read.csv(test_path, check.names = TRUE)
  
  cat("訓練資料維度:", dim(train_data), "\n")
  cat("測試資料維度:", dim(test_data), "\n")
  
  # 檢查目標變數
  stopifnot(target_col %in% names(train_data))
  stopifnot(target_col %in% names(test_data))
  
  train_data[[target_col]] <- factor(train_data[[target_col]])
  test_data[[target_col]] <- factor(test_data[[target_col]])
  
  cat("訓練集類別分佈:\n")
  print(table(train_data[[target_col]]))
  cat("測試集類別分佈:\n")
  print(table(test_data[[target_col]]))
  
  # 清理資料
  train_data <- train_data %>% select(-any_of(c("File", "X", "")))
  test_data <- test_data %>% select(-any_of(c("File", "X", "")))
  
  # 獲取特徵欄位
  num_cols_train <- sapply(train_data, is.numeric)
  num_cols_test <- sapply(test_data, is.numeric)
  
  feature_cols_train <- setdiff(names(train_data)[num_cols_train], target_col)
  feature_cols_test <- setdiff(names(test_data)[num_cols_test], target_col)
  
  # 確保訓練和測試集有相同的特徵
  common_features <- intersect(feature_cols_train, feature_cols_test)
  
  if (length(common_features) == 0) {
    stop("訓練集和測試集沒有共同的特徵")
  }
  
  cat("共同特徵數:", length(common_features), "\n")
  
  # 分析特徵類型
  feature_types <- list(
    HOG_PCA = sum(grepl("^HOG_PC", common_features)),
    LBP_PCA = sum(grepl("^LBP_PC", common_features)),
    IMG = sum(grepl("^IMG__", common_features)),
    TEXSH = sum(grepl("^TEXSH__", common_features)),
    CORNER = sum(grepl("^CORNER__", common_features))
  )
  
  cat("特徵類型分布:\n")
  for (type_name in names(feature_types)) {
    if (feature_types[[type_name]] > 0) {
      cat(sprintf("  %s: %d\n", type_name, feature_types[[type_name]]))
    }
  }
  
  # 只保留共同特徵和目標變數
  train_data <- train_data %>% select(all_of(c(target_col, common_features)))
  test_data <- test_data %>% select(all_of(c(target_col, common_features)))
  
  # 填補 NA (Random Forest 對 NA 較敏感)
  for(col in common_features){
    if(any(is.na(train_data[[col]]))) {
      train_data[[col]][is.na(train_data[[col]])] <- median(train_data[[col]], na.rm = TRUE)
    }
    if(any(is.na(test_data[[col]]))) {
      test_data[[col]][is.na(test_data[[col]])] <- median(test_data[[col]], na.rm = TRUE)
    }
  }
  
  # 設定預設的 mtry 參數
  if (is.null(rf_mtry)) {
    rf_mtry <- max(1, floor(sqrt(length(common_features))))
  }
  
  cat("Random Forest 參數:\n")
  cat("  ntree:", rf_ntree, "\n")
  cat("  mtry:", rf_mtry, "\n")
  cat("  nodesize:", rf_nodesize, "\n")
  if (!is.null(rf_maxnodes)) cat("  maxnodes:", rf_maxnodes, "\n")
  
  # 2. 在訓練集內進行 5 折分層 CV
  cat("\n步驟 2: 在訓練集內進行 5 折分層 CV...\n")
  cat("直接對訓練集 (", nrow(train_data), "樣本) 進行5折分層交叉驗證\n")
  
  # 建立分層 CV 折 (使用 seed = 789 保持一致性)
  cv_fold_ids <- make_stratified_folds(train_data[[target_col]], k = cv_folds, seed = 789)
  
  cat("各折類別分布驗證:\n")
  for (i in seq_len(cv_folds)) {
    val_samples <- length(cv_fold_ids[[i]])
    train_samples <- nrow(train_data) - val_samples
    
    cat(sprintf("Fold %d - 驗證集 (%d 樣本): ", i, val_samples))
    val_dist <- table(train_data[cv_fold_ids[[i]], target_col])
    cat(paste(names(val_dist), val_dist, sep="=", collapse=", "), "\n")
    
    cat(sprintf("Fold %d - 訓練集 (%d 樣本): ", i, train_samples))
    train_idx <- setdiff(seq_len(nrow(train_data)), cv_fold_ids[[i]])
    train_dist <- table(train_data[train_idx, target_col])
    cat(paste(names(train_dist), train_dist, sep="=", collapse=", "), "\n\n")
  }
  
  cv_results <- data.frame()
  cv_predictions <- data.frame()
  cv_class_metrics_list <- list()
  
  for (fold in seq_len(cv_folds)) {
    cat(sprintf("執行 CV Fold %d/%d...\n", fold, cv_folds))
    
    val_idx <- cv_fold_ids[[fold]]
    tr_idx <- setdiff(seq_len(nrow(train_data)), val_idx)
    
    cv_train <- train_data[tr_idx, ]
    cv_val <- train_data[val_idx, ]
    
    cat(sprintf("  CV 訓練: %d 樣本, CV 驗證: %d 樣本\n", nrow(cv_train), nrow(cv_val)))
    
    # 訓練 Random Forest
    formula <- as.formula(paste("Label ~", paste(common_features, collapse = " + ")))
    rf_result <- safe_rf_train(
      formula = formula,
      data = cv_train,
      ntree = rf_ntree,
      mtry = rf_mtry,
      nodesize = rf_nodesize,
      maxnodes = rf_maxnodes
    )
    
    if (rf_result$success) {
      # 預測
      cv_pred <- predict(rf_result$model, cv_val)
      
      # 計算多類別指標
      fold_metrics <- calculate_multiclass_metrics(cv_val[[target_col]], cv_pred)
      
      # 保存預測結果
      fold_predictions <- data.frame(
        Fold = fold,
        True_Label = cv_val[[target_col]],
        Predicted_Label = cv_pred,
        Correct = cv_val[[target_col]] == cv_pred,
        stringsAsFactors = FALSE
      )
      cv_predictions <- rbind(cv_predictions, fold_predictions)
      
      # 保存各類別指標
      fold_class_metrics <- fold_metrics$class_metrics
      fold_class_metrics$Fold <- fold
      cv_class_metrics_list[[fold]] <- fold_class_metrics
      
      # 保存整體指標
      fold_result <- data.frame(
        Fold = fold,
        Overall_Accuracy = fold_metrics$overall_accuracy,
        Macro_Precision = fold_metrics$macro_precision,
        Macro_Recall = fold_metrics$macro_recall,
        Macro_F1 = fold_metrics$macro_f1,
        Weighted_Precision = fold_metrics$weighted_precision,
        Weighted_Recall = fold_metrics$weighted_recall,
        Weighted_F1 = fold_metrics$weighted_f1,
        Balanced_Accuracy = fold_metrics$macro_balanced_accuracy,
        Confusion_Matrix = I(list(fold_metrics$confusion_matrix)),
        stringsAsFactors = FALSE
      )
      cv_results <- rbind(cv_results, fold_result)
      
      cat(sprintf("  結果 - 準確度: %.4f, Macro-F1: %.4f, Weighted-F1: %.4f\n", 
                  fold_metrics$overall_accuracy, fold_metrics$macro_f1, fold_metrics$weighted_f1))
    } else {
      cat(sprintf("  Fold %d Random Forest 訓練失敗: %s\n", fold, rf_result$error))
    }
  }
  
  # 3. 最終模型訓練和測試
  cat("\n步驟 3: 最終模型訓練和測試...\n")
  cat("使用全部訓練集訓練最終模型，然後在測試集上評估\n")
  
  # 使用全部訓練集訓練最終模型
  formula <- as.formula(paste("Label ~", paste(common_features, collapse = " + ")))
  
  final_rf_result <- safe_rf_train(
    formula = formula,
    data = train_data,
    ntree = rf_ntree,
    mtry = rf_mtry,
    nodesize = rf_nodesize,
    maxnodes = rf_maxnodes
  )
  
  if (!final_rf_result$success) {
    stop("最終模型訓練失敗: ", final_rf_result$error)
  }
  
  final_model <- final_rf_result$model
  
  # 預測
  train_pred <- predict(final_model, train_data)
  test_pred <- predict(final_model, test_data)
  
  # 計算最終指標
  train_metrics <- calculate_multiclass_metrics(train_data[[target_col]], train_pred)
  test_metrics <- calculate_multiclass_metrics(test_data[[target_col]], test_pred)
  
  # 4. 結果報告
  cat("\n", rep("=", 80), "\n")
  cat("基於已篩選特徵的 5折分層 CV Random Forest 分析結果\n")
  cat(rep("=", 80), "\n")
  
  cat("模型配置:\n")
  cat("  訓練檔案:", train_path, "\n")
  cat("  測試檔案:", test_path, "\n")
  cat("  使用特徵數:", length(common_features), "\n")
  cat("  RF ntree:", rf_ntree, "\n")
  cat("  RF mtry:", rf_mtry, "\n")
  cat("  RF nodesize:", rf_nodesize, "\n")
  if (!is.null(rf_maxnodes)) cat("  RF maxnodes:", rf_maxnodes, "\n")
  cat("  CV 折數:", cv_folds, "\n")
  
  cat("\n資料統計:\n")
  cat("  訓練樣本數:", nrow(train_data), "\n")
  cat("  測試樣本數:", nrow(test_data), "\n")
  cat("  總特徵數:", length(common_features), "\n")
  cat("  類別數:", nlevels(train_data[[target_col]]), "\n")
  
  # CV 結果摘要
  if (nrow(cv_results) > 0) {
    cat("\n交叉驗證結果摘要 (", cv_folds, "折平均):\n")
    cat("  CV 準確度: ", sprintf("%.4f ± %.4f", mean(cv_results$Overall_Accuracy), sd(cv_results$Overall_Accuracy)), "\n")
    cat("  CV Macro-Precision: ", sprintf("%.4f ± %.4f", mean(cv_results$Macro_Precision), sd(cv_results$Macro_Precision)), "\n")
    cat("  CV Macro-Recall: ", sprintf("%.4f ± %.4f", mean(cv_results$Macro_Recall), sd(cv_results$Macro_Recall)), "\n")
    cat("  CV Macro-F1: ", sprintf("%.4f ± %.4f", mean(cv_results$Macro_F1), sd(cv_results$Macro_F1)), "\n")
    cat("  CV Weighted-F1: ", sprintf("%.4f ± %.4f", mean(cv_results$Weighted_F1), sd(cv_results$Weighted_F1)), "\n")
    cat("  CV Balanced-Accuracy: ", sprintf("%.4f ± %.4f", mean(cv_results$Balanced_Accuracy), sd(cv_results$Balanced_Accuracy)), "\n")
  }
  
  cat("\n最終模型性能:\n")
  cat("  訓練集準確度:", sprintf("%.4f", train_metrics$overall_accuracy), "\n")
  cat("  測試集準確度:", sprintf("%.4f", test_metrics$overall_accuracy), "\n")
  
  overfitting_gap <- train_metrics$overall_accuracy - test_metrics$overall_accuracy
  cat("  過擬合程度:", sprintf("%.4f", overfitting_gap), "\n")
  
  # 檢查過度配適警告
  if (train_metrics$overall_accuracy >= 0.98) {
    cat("\n⚠️ 過度配適警告:\n")
    cat("  訓練集準確度過高 (≥0.98)，建議調整參數:\n")
    cat("  - 增加 nodesize (目前:", rf_nodesize, ")\n")
    cat("  - 減少 ntree (目前:", rf_ntree, ")\n")
    cat("  - 設定 maxnodes 限制樹的複雜度\n")
    cat("  - 減少 mtry (目前:", rf_mtry, ")\n")
  }
  
  cat("\n測試集多類別指標:\n")
  cat("  Macro-Precision:", sprintf("%.4f", test_metrics$macro_precision), "\n")
  cat("  Macro-Recall:", sprintf("%.4f", test_metrics$macro_recall), "\n")
  cat("  Macro-F1:", sprintf("%.4f", test_metrics$macro_f1), "\n")
  cat("  Weighted-F1:", sprintf("%.4f", test_metrics$weighted_f1), "\n")  
  cat("  Balanced-Accuracy:", sprintf("%.4f", test_metrics$macro_balanced_accuracy), "\n")
  
  cat("\n各類別詳細指標 (測試集):\n")
  print(test_metrics$class_metrics)
  
  # 特徵重要性分析
  if (!is.null(final_model$importance)) {
    cat("\n特徵重要性 (前10名):\n")
    importance_df <- data.frame(
      Feature = rownames(final_model$importance),
      Importance = final_model$importance[,1]
    )
    importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
    print(head(importance_df, 10))
  }
  
  # 5. 視覺化
  cat("\n步驟 4: 生成視覺化...\n")
  
  if (save_results && !dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # 圖1: 各類別指標比較 (長條圖)
  test_class_metrics <- test_metrics$class_metrics
  class_metrics_long <- reshape2::melt(test_class_metrics, 
                                       id.vars = "Class", 
                                       variable.name = "Metric", 
                                       value.name = "Score")
  
  p1 <- ggplot(class_metrics_long, aes(x = Class, y = Score, fill = Metric)) +
    geom_col(position = "dodge", alpha = 0.8) +
    geom_text(aes(label = sprintf("%.3f", Score)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.3, size = 3) +
    labs(title = "各類別預測指標比較 (測試集) - Random Forest",
         x = "類別", y = "分數") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(size = 14, face = "bold")) +
    ylim(0, 1.1)
  
  # 圖2: 整體指標摘要
  overall_metrics <- data.frame(
    Metric = c("Overall Accuracy", "Macro-F1", "Weighted-F1", "Balanced-Accuracy"),
    Score = c(test_metrics$overall_accuracy, test_metrics$macro_f1, 
              test_metrics$weighted_f1, test_metrics$macro_balanced_accuracy),
    stringsAsFactors = FALSE
  )
  
  p2 <- ggplot(overall_metrics, aes(x = Metric, y = Score)) +
    geom_col(fill = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"), alpha = 0.8) +
    geom_text(aes(label = sprintf("%.4f", Score)), vjust = -0.3, size = 5, fontface = "bold") +
    labs(title = "整體 Random Forest 指標摘要 (測試集)",
         subtitle = paste("Overall Accuracy:", sprintf("%.4f", test_metrics$overall_accuracy),
                          "| Macro-F1:", sprintf("%.4f", test_metrics$macro_f1),
                          "| Weighted-F1:", sprintf("%.4f", test_metrics$weighted_f1)),
         x = "指標類型", y = "分數") +
    theme_minimal() +
    theme(plot.title = element_text(size = 14, face = "bold"),
          plot.subtitle = element_text(size = 12, color = "darkgreen"),
          axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(0, 1.1)
  
  # 圖3: CV 結果分布
  if (nrow(cv_results) > 0) {
    cv_metrics_long <- reshape2::melt(cv_results[, c("Fold", "Overall_Accuracy", "Macro_F1", "Weighted_F1", "Balanced_Accuracy")],
                                      id.vars = "Fold",
                                      variable.name = "Metric",
                                      value.name = "Score")
    
    p3 <- ggplot(cv_metrics_long, aes(x = factor(Fold), y = Score, fill = Metric)) +
      geom_col(position = "dodge", alpha = 0.8) +
      geom_text(aes(label = sprintf("%.3f", Score)), 
                position = position_dodge(width = 0.9), 
                vjust = -0.3, size = 2.5) +
      labs(title = "5折交叉驗證結果分析 - Random Forest",
           x = "CV 折數", y = "分數") +
      theme_minimal() +
      theme(plot.title = element_text(size = 14, face = "bold")) +
      ylim(0, 1.1)
  }
  
  # 圖4: 特徵重要性圖
  p6 <- NULL
  if (!is.null(final_model$importance)) {
    importance_df <- data.frame(
      Feature = rownames(final_model$importance),
      Importance = final_model$importance[,1]
    )
    importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
    top_features <- head(importance_df, 15)
    
    p6 <- ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_col(fill = "forestgreen", alpha = 0.7) +
      geom_text(aes(label = sprintf("%.1f", Importance)), hjust = -0.1, size = 3) +
      coord_flip() +
      labs(title = "特徵重要性排序 (前15名)",
           subtitle = "基於 Random Forest Mean Decrease in Node Impurity",
           x = "特徵", y = "重要性分數") +
      theme_minimal() +
      theme(plot.title = element_text(size = 14, face = "bold"),
            plot.subtitle = element_text(size = 10, color = "darkgreen"))
  }
  
  # 混淆矩陣可視化函數
  plot_confusion_matrix <- function(conf_matrix, title) {
    cm_table <- as.table(conf_matrix$table)
    cm_df <- as.data.frame(cm_table)
    names(cm_df) <- c("Predicted", "Reference", "Freq")
    
    total_samples <- sum(cm_df$Freq)
    cm_df$Percentage <- round(cm_df$Freq / total_samples * 100, 1)
    
    ggplot(cm_df, aes(x = Reference, y = Predicted, fill = Freq)) +
      geom_tile(color = "white", linewidth = 1) +
      geom_text(aes(label = Freq), 
                color = ifelse(cm_df$Freq > max(cm_df$Freq)/2, "white", "black"),
                size = 4, fontface = "bold") +
      scale_fill_gradient(low = "#f7f7f7", high = "#228B22", name = "Count") +
      labs(title = title, x = "真實標籤", y = "預測標籤") +
      theme_minimal() +
      theme(plot.title = element_text(size = 12, face = "bold", hjust = 0.5)) +
      coord_fixed()
  }
  
  # 訓練集和測試集混淆矩陣
  p4 <- plot_confusion_matrix(train_metrics$confusion_matrix, 
                              sprintf("訓練集混淆矩陣 (準確度: %.3f)", train_metrics$overall_accuracy))
  p5 <- plot_confusion_matrix(test_metrics$confusion_matrix, 
                              sprintf("測試集混淆矩陣 (準確度: %.3f)", test_metrics$overall_accuracy))
  
  # CV 各折混淆矩陣
  cv_cm_plots <- list()
  if (nrow(cv_results) > 0) {
    cat("生成CV各折混淆矩陣圖...\n")
    for (fold in 1:nrow(cv_results)) {
      if (is.list(cv_results$Confusion_Matrix) && length(cv_results$Confusion_Matrix) >= fold) {
        fold_cm <- cv_results$Confusion_Matrix[[fold]]
        if (!is.null(fold_cm)) {
          fold_accuracy <- cv_results$Overall_Accuracy[fold]
          fold_macro_f1 <- cv_results$Macro_F1[fold]
          
          cm_title <- sprintf("CV Fold %d (準確度: %.3f, Macro-F1: %.3f)", 
                              fold, fold_accuracy, fold_macro_f1)
          
          cv_cm_plots[[fold]] <- plot_confusion_matrix(fold_cm, cm_title)
          cat(sprintf("  生成 Fold %d 混淆矩陣\n", fold))
        }
      }
    }
  }
  
  # 保存圖表
  if (save_results) {
    cat("保存視覺化圖表...\n")
    
    ggsave(file.path(output_dir, "rf_class_metrics_comparison.png"), p1, width = 12, height = 8, dpi = 300)
    cat("  已保存: rf_class_metrics_comparison.png\n")
    
    ggsave(file.path(output_dir, "rf_overall_metrics_summary.png"), p2, width = 12, height = 6, dpi = 300)
    cat("  已保存: rf_overall_metrics_summary.png\n")
    
    if (exists("p3")) {
      ggsave(file.path(output_dir, "rf_cv_results_distribution.png"), p3, width = 12, height = 6, dpi = 300)
      cat("  已保存: rf_cv_results_distribution.png\n")
    }
    
    ggsave(file.path(output_dir, "rf_train_confusion_matrix.png"), p4, width = 8, height = 6, dpi = 300)
    cat("  已保存: rf_train_confusion_matrix.png\n")
    
    ggsave(file.path(output_dir, "rf_test_confusion_matrix.png"), p5, width = 8, height = 6, dpi = 300)
    cat("  已保存: rf_test_confusion_matrix.png\n")
    
    # 保存特徵重要性圖
    if (!is.null(p6)) {
      ggsave(file.path(output_dir, "rf_feature_importance.png"), p6, width = 10, height = 8, dpi = 300)
      cat("  已保存: rf_feature_importance.png\n")
    }
    
    # 保存每折的混淆矩陣
    if (length(cv_cm_plots) > 0) {
      for (fold in 1:length(cv_cm_plots)) {
        if (!is.null(cv_cm_plots[[fold]])) {
          tryCatch({
            filename <- sprintf("rf_cv_fold_%d_confusion_matrix.png", fold)
            ggsave(file.path(output_dir, filename), cv_cm_plots[[fold]], 
                   width = 6, height = 5, dpi = 300)
            cat(sprintf("  已保存: %s\n", filename))
          }, error = function(e) {
            cat(sprintf("  保存 Fold %d 失敗: %s\n", fold, e$message))
          })
        }
      }
    }
  }
  
  # 6. 保存結果
  if (save_results) {
    cat("\n步驟 5: 保存結果...\n")
    
    # 保存模型摘要
    model_summary <- data.frame(
      Parameter = c("Train_File", "Test_File", "Feature_Count", "RF_ntree", 
                    "RF_mtry", "RF_nodesize", "RF_maxnodes", "CV_Folds",
                    "Train_Samples", "Test_Samples", "Classes",
                    "CV_Accuracy_Mean", "CV_Accuracy_Std", "CV_MacroF1_Mean", "CV_MacroF1_Std",
                    "CV_WeightedF1_Mean", "CV_WeightedF1_Std", "CV_BalancedAcc_Mean", "CV_BalancedAcc_Std",
                    "Train_Accuracy", "Test_Accuracy", "Overfitting_Gap",
                    "Test_MacroF1", "Test_WeightedF1", "Test_BalancedAccuracy"),
      Value = c(train_path, test_path, length(common_features), rf_ntree,
                rf_mtry, rf_nodesize, ifelse(is.null(rf_maxnodes), "NULL", rf_maxnodes), cv_folds,
                nrow(train_data), nrow(test_data), nlevels(train_data[[target_col]]),
                ifelse(nrow(cv_results) > 0, sprintf("%.4f", mean(cv_results$Overall_Accuracy)), "NA"),
                ifelse(nrow(cv_results) > 0, sprintf("%.4f", sd(cv_results$Overall_Accuracy)), "NA"),
                ifelse(nrow(cv_results) > 0, sprintf("%.4f", mean(cv_results$Macro_F1)), "NA"),
                ifelse(nrow(cv_results) > 0, sprintf("%.4f", sd(cv_results$Macro_F1)), "NA"),
                ifelse(nrow(cv_results) > 0, sprintf("%.4f", mean(cv_results$Weighted_F1)), "NA"),
                ifelse(nrow(cv_results) > 0, sprintf("%.4f", sd(cv_results$Weighted_F1)), "NA"),
                ifelse(nrow(cv_results) > 0, sprintf("%.4f", mean(cv_results$Balanced_Accuracy)), "NA"),
                ifelse(nrow(cv_results) > 0, sprintf("%.4f", sd(cv_results$Balanced_Accuracy)), "NA"),
                sprintf("%.4f", train_metrics$overall_accuracy),
                sprintf("%.4f", test_metrics$overall_accuracy),
                sprintf("%.4f", overfitting_gap),
                sprintf("%.4f", test_metrics$macro_f1),
                sprintf("%.4f", test_metrics$weighted_f1),
                sprintf("%.4f", test_metrics$macro_balanced_accuracy)),
      stringsAsFactors = FALSE
    )
    write.csv(model_summary, file.path(output_dir, "rf_model_summary.csv"), row.names = FALSE)
    
    # 保存特徵重要性
    if (!is.null(final_model$importance)) {
      importance_df <- data.frame(
        Feature = rownames(final_model$importance),
        Importance = final_model$importance[,1]
      )
      importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
      write.csv(importance_df, file.path(output_dir, "rf_feature_importance.csv"), row.names = FALSE)
    }
    
    # 保存特徵清單和其他結果檔案
    write.csv(data.frame(Feature = common_features), 
              file.path(output_dir, "used_features.csv"), row.names = FALSE)
    
    write.csv(train_metrics$class_metrics, 
              file.path(output_dir, "train_class_metrics.csv"), row.names = FALSE)
    write.csv(test_metrics$class_metrics, 
              file.path(output_dir, "test_class_metrics.csv"), row.names = FALSE)
    
    if (nrow(cv_results) > 0) {
      write.csv(cv_results, file.path(output_dir, "cv_detailed_results.csv"), row.names = FALSE)
      write.csv(cv_predictions, file.path(output_dir, "cv_predictions.csv"), row.names = FALSE)
      
      if (length(cv_class_metrics_list) > 0) {
        all_cv_class_metrics <- do.call(rbind, cv_class_metrics_list)
        write.csv(all_cv_class_metrics, file.path(output_dir, "cv_class_metrics_by_fold.csv"), row.names = FALSE)
      }
    }
    
    # 保存預測結果
    train_predictions_df <- data.frame(
      True_Label = train_data[[target_col]],
      Predicted_Label = train_pred,
      Correct = train_data[[target_col]] == train_pred,
      stringsAsFactors = FALSE
    )
    write.csv(train_predictions_df, file.path(output_dir, "final_train_predictions.csv"), row.names = FALSE)
    
    test_predictions_df <- data.frame(
      True_Label = test_data[[target_col]],
      Predicted_Label = test_pred,
      Correct = test_data[[target_col]] == test_pred,
      stringsAsFactors = FALSE
    )
    write.csv(test_predictions_df, file.path(output_dir, "final_test_predictions.csv"), row.names = FALSE)
  }
  
  total_time <- as.numeric(difftime(Sys.time(), analysis_start_time, units = "mins"))
  
  cat("\n", rep("=", 80), "\n")
  cat("基於已篩選特徵的 5折分層 CV Random Forest 分析完成！\n")
  cat(rep("=", 80), "\n")
  cat("總執行時間:", sprintf("%.1f", total_time), "分鐘\n")
  
  if (save_results) {
    cat("\n檔案已保存至:", output_dir, "\n")
    cat("主要檔案:\n")
    cat("  - rf_model_summary.csv: 完整模型摘要\n")
    cat("  - rf_feature_importance.csv: 特徵重要性排序\n")
    cat("  - *_class_metrics.csv: 各類別詳細指標\n")
    cat("  - cv_detailed_results.csv: CV 詳細結果\n")
    cat("  - final_*_predictions.csv: 最終預測結果\n")
    cat("  - *.png: 視覺化圖表 (包含特徵重要性圖)\n")
  }
  
  # 7. 返回結果
  return(list(
    # 模型和資料
    final_model = final_model,
    used_features = common_features,
    train_data = train_data,
    test_data = test_data,
    
    # CV 結果
    cv_results = cv_results,
    cv_predictions = cv_predictions,
    cv_class_metrics = if(length(cv_class_metrics_list) > 0) do.call(rbind, cv_class_metrics_list) else NULL,
    
    # 最終評估結果
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    train_predictions = train_pred,
    test_predictions = test_pred,
    
    # 特徵重要性
    feature_importance = if(!is.null(final_model$importance)) {
      df <- data.frame(
        Feature = rownames(final_model$importance),
        Importance = final_model$importance[,1]
      )
      df[order(df$Importance, decreasing = TRUE), ]
    } else NULL,
    
    # 視覺化圖表
    plots = list(
      class_metrics_comparison = p1,
      overall_metrics_summary = p2,
      cv_results_distribution = if(exists("p3")) p3 else NULL,
      train_confusion_matrix = p4,
      test_confusion_matrix = p5,
      feature_importance = if(!is.null(p6)) p6 else NULL,
      cv_confusion_matrices = if(length(cv_cm_plots) > 0) cv_cm_plots else NULL
    ),
    
    # 模型配置
    model_config = list(
      train_file = train_path,
      test_file = test_path,
      rf_ntree = rf_ntree,
      rf_mtry = rf_mtry,
      rf_nodesize = rf_nodesize,
      rf_maxnodes = rf_maxnodes,
      n_features = length(common_features),
      cv_folds = cv_folds
    ),
    
    # 特徵類型分析
    feature_types = feature_types,
    
    # 執行時間
    execution_time_minutes = total_time
  ))
}

# ========= 快速執行函數 =========

# 快速執行 Random Forest 分析（使用預設參數）
quick_rf_separate_analysis <- function() {
  cat("=== 快速執行 Random Forest 分析 (分離檔案版) ===\n")
  cat("讀取檔案: train_final_variance_20.csv & test_final_variance_20.csv\n")
  cat("使用預設參數: ntree=500, nodesize=10, mtry=sqrt(特徵數)\n")
  cat("5折分層交叉驗證\n\n")
  
  result <- run_rf_separate_files_analysis(
    train_path = "C:/Users/yu/OneDrive/桌面/train_final_XBG_20.csv",
    test_path = "C:/Users/yu/OneDrive/桌面/test_final_XGB_20.csv",
    rf_ntree = 500,
    rf_mtry = NULL,  # 自動計算
    rf_nodesize = 10,
    rf_maxnodes = NULL,
    cv_folds = 5,
    save_results = TRUE
  )
  
  # 結果摘要
  cat("\n=== 結果摘要 ===\n")
  if (nrow(result$cv_results) > 0) {
    cat("CV 平均準確度:", sprintf("%.4f ± %.4f", 
                             mean(result$cv_results$Overall_Accuracy), 
                             sd(result$cv_results$Overall_Accuracy)), "\n")
    cat("CV 平均 Macro-F1:", sprintf("%.4f ± %.4f", 
                                   mean(result$cv_results$Macro_F1), 
                                   sd(result$cv_results$Macro_F1)), "\n")
  }
  cat("測試集準確度:", sprintf("%.4f", result$test_metrics$overall_accuracy), "\n")
  cat("測試集 Macro-F1:", sprintf("%.4f", result$test_metrics$macro_f1), "\n")
  cat("測試集 Weighted-F1:", sprintf("%.4f", result$test_metrics$weighted_f1), "\n")
  
  # 過度配適檢查
  overfitting_gap <- result$train_metrics$overall_accuracy - result$test_metrics$overall_accuracy
  if (result$train_metrics$overall_accuracy >= 0.98) {
    cat("\n⚠️ 過度配適警告: 訓練集準確度過高，建議調整參數\n")
  }
  cat("過擬合程度:", sprintf("%.4f", overfitting_gap), "\n")
  
  return(result)
}

# 調整過度配適的 Random Forest 分析
tuned_rf_separate_analysis <- function() {
  cat("=== 調整後的 Random Forest 分析 (分離檔案版) ===\n")
  cat("讀取檔案: train_final_variance_20.csv & test_final_variance_20.csv\n")
  cat("使用調整參數: ntree=300, nodesize=20, maxnodes=50\n")
  cat("5折分層交叉驗證\n\n")
  
  result <- run_rf_separate_files_analysis(
    train_path = "C:/Users/yu/OneDrive/桌面/train_final_XBG_20.csv",
    test_path = "C:/Users/yu/OneDrive/桌面/test_final_XGB_20.csv",
    rf_ntree = 300,
    rf_mtry = NULL,
    rf_nodesize = 20,  # 增加最小節點樣本數
    rf_maxnodes = 50,  # 限制最大節點數
    cv_folds = 5,
    save_results = TRUE,
    output_dir = "rf_tuned_results"
  )
  
  return(result)
}

# 複製整個程式碼到R中執行，然後運行：
rf_results <- quick_rf_separate_analysis()
