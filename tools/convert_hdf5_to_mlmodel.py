import coremltools

# import tfcoreml
# from keras.models import load_model

# coremltools.converters.tensorflow


coreml_model = coremltools.converters.keras.convert(
    # coreml_model = coremltools.converters.tensorflow(
    "../model/artist-model_3_30.hdf5",
    # input_names="image",
    # image_input_names="image",
    # output_names="Prediction",
    # class_labels=["Pierre_Auguste_Renoir", "Salvador_Dali", "Rembrandt"],
)

coreml_model.save("../model/artist-model_3_30.mlmodel")
# coreml_model = coremltools.converters.keras.convert('my_keras_model.h5')
