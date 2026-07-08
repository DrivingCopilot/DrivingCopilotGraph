from services.few_shots_examples import few_shot_examples
from graph.vector_services import store_few_shot_examples
import logging

logging.basicConfig(level=logging.INFO)
store_few_shot_examples(few_shot_examples)
