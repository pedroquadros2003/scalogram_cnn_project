import logging
logger = logging.getLogger(__name__)

import keras_tuner as kt
import os
import gc
import tensorflow as tf 


class CustomTuner(kt.RandomSearch):

    def __init__(self, input_folder, output_folder, model_hyper_params, model_train_params, run_model, create_model, *args, **kwargs):

        self.input_folder  = input_folder
        self.output_folder = output_folder
        self.model_hyper_params = model_hyper_params
        self.model_train_params = model_train_params
        self.create_model = create_model
        self.run_model = run_model
        super().__init__(*args, **kwargs)


    # ======================================
    # IMPLEMENT SAVE_MODEL
    # ======================================

    def save_model(self, trial_id, model):

        model_dir = os.path.join(
            self.project_dir,
            "saved_models",
            f"trial_{trial_id}"
        )

        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.keras")

        model.save(model_path)


    def run_trial(self, trial):

        hp = trial.hyperparameters

        params = {}

        # Create Model ID
        model_id = f"model_{trial.trial_id}"
        params["model_id"] = model_id

        ## Add to params the parameters that will be optimized by Keras Tuner
        params.update(hp.values)

        ## Add to params the fixed parameters
        def inject_fixed(params, space):
            for key, cfg in space.items():
                if cfg["mode"] == "fixed":
                    params[key] = cfg["values"][0]

        inject_fixed(params, self.model_hyper_params)
        inject_fixed(params, self.model_train_params)

        logger.debug("Trial params: %s", params)
        

        # ======================================
        # CREATE MODEL
        # ======================================

        model, callback = self.create_model(params)

        # ======================================
        # RUN TRAINING
        # ======================================

        try:

            _, history = self.run_model(
                model=model,
                callback=callback,
                parameters=params,
                input_folder=self.input_folder,
                output_folder=self.output_folder,
            )

            val_loss = history.history["val_loss"][-1]

        except Exception as e:

            logger.error("Trial failed: %s", e)

            val_loss = None

        # ======================================
        # REPORT RESULT
        # ======================================

        if val_loss is None:
            val_loss = float("inf")

        self.oracle.update_trial(
            trial.trial_id,
            {"val_loss": val_loss},
        )

        self.save_model(trial.trial_id, model)

        tf.keras.backend.clear_session()
        gc.collect()
