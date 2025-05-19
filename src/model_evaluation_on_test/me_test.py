from Mylib import myfuncs, myclasses
import os


def load_data_for_model_evaluation_on_test(
    test_data_path,
    correction_transformer_path,
    transformation_transformer_path,
    model_path,
):
    test_data = myfuncs.load_python_object(test_data_path)
    correction_transformer = myfuncs.load_python_object(
        os.path.join(correction_transformer_path)
    )
    transformation_transformer = myfuncs.load_python_object(
        os.path.join(transformation_transformer_path)
    )
    model = myfuncs.load_python_object(model_path)

    return (
        test_data,
        correction_transformer,
        transformation_transformer,
        model,
    )


def transform_test_data(test_data, correction_transformer, transformation_transformer):
    test_data_corrected = correction_transformer.transform(test_data)
    test_data_transformed = transformation_transformer.transform(test_data_corrected)
    target_col = myfuncs.get_target_col_from_df_26(test_data_transformed)
    test_features = test_data_transformed.drop(columns=[target_col])
    test_target = test_data_transformed[target_col]

    return test_features, test_target


def evaluate_model_on_test(
    test_features,
    test_target,
    model,
    model_name,
    model_index,
    scoring,
    root_dir,
):
    print(
        f"\n========Bắt đầu đánh giá model index {model_name} - {model_index} !!!!!===================\n"
    )

    final_model_results_text = "===========Kết quả đánh giá model ================\n"

    ## Chỉ số scoring
    test_score = myfuncs.evaluate_model_on_one_scoring_17(
        model, test_features, test_target, scoring
    )
    final_model_results_text += f"====Chỉ số scoring====\n"
    final_model_results_text += f"Test {scoring}: {test_score}\n"

    # Các chỉ số khác: rmse + mae
    final_model_results_text += "====Các chỉ số khác===========\n"
    model_results_text, test_confusion_matrix = myclasses.RegressorEvaluator(
        model=model,
        train_feature_data=test_features,
        train_target_data=test_target,
    ).evaluate()
    final_model_results_text += model_results_text

    # Lưu vào file results.txt
    with open(os.path.join(root_dir, "results.txt"), mode="w") as file:
        file.write(model_results_text)
