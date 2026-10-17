"""One user-reflection intervention; item, selection, and ranking prompts stay fixed."""
from copy import deepcopy


def neutral_user_feedback(prompts):
    result = deepcopy(prompts)
    context = ("Recently, the user consumed '$pos_item_title'. The item '$neg_item_title' was sampled "
               "as an unobserved comparison. There is no observation that the user saw, rejected, "
               "or disliked that comparison item.\nCandidate descriptions:\n$list_of_item_description\n"
               "The previous system predicted a choice and gave this explanation:\n$system_reason\n"
               "That explanation is a model prediction, not additional user feedback.\n")
    task = ("Update the user's self-introduction by integrating the preferences suggested by the observed "
            "consumption. Preserve past preferences unless the observed feedback contradicts them. "
            "Do not infer dislike of the unobserved comparison or its features from missing interaction. "
            "Describe new preferences first, then retained past preferences, and retain dislikes only "
            "if the previous memory already provides evidence for them.\n")
    for key in ("user_prompt_template", "user_prompt_template_true"):
        # Preserve the author's formatting, 180-word limit, and personalization notes.
        notes = prompts[key].split("Important note:", 1)[1]
        result[key] = context + task + "\nImportant note:" + notes
    return result
