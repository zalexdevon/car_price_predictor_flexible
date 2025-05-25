from Mylib import myfuncs, myclasses
import os


def load_data_for_me_trainval(data_transformation_path, model_path):
    train_features = myfuncs.load_python_object(
        os.path.join(data_transformation_path), "train_features.pkl"
    )
    train_target = myfuncs.load_python_object(
        os.path.join(data_transformation_path), "train_target.pkl"
    )
    val_features = myfuncs.load_python_object(
        os.path.join(data_transformation_path), "val_features.pkl"
    )
    val_target = myfuncs.load_python_object(
        os.path.join(data_transformation_path), "val_target.pkl"
    )

    model = myfuncs.load_python_object(model_path)

    return train_features, train_target, val_features, val_target, model


def evaluate_model_on_train_val(
    train_features,
    train_target,
    val_features,
    val_target,
    model,
    root_dir,
):
    final_model_results_text = (
        "===============Kết quả đánh giá model==================\n"
    )
    model_results_text = myclasses.RegressorEvaluator(
        model=model,
        train_feature_data=train_features,
        train_target_data=train_target,
        val_feature_data=val_features,
        val_target_data=val_target,
    ).evaluate()
    final_model_results_text += model_results_text

    # Lưu vào file results.txt
    with open(os.path.join(root_dir, "result.txt"), mode="w") as file:
        file.write(final_model_results_text)
