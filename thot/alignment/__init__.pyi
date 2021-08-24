from typing import Sequence, Tuple, overload


class AlignmentModel():
    @property
    def num_sentence_pairs(self) -> int: ...

    @property
    def src_vocab_size(self) -> int: ...

    @property
    def trg_vocab_size(self) -> int: ...
    
    @property
    def variational_bayes(self) -> bool: ...
    
    @variational_bayes.setter
    def variational_bayes(self, arg1: bool) -> None: ...

    def add_sentence_pair(self, src_sentence: Sequence[str], trg_sentence: Sequence[str], count: float) -> Tuple[int, int]: ...
    def get_sentence_pair(self, n: int) -> Tuple[Sequence[str], Sequence[str], float]: ...
    
    def add_src_word(self, word: str) -> int: ...
    def get_src_word(self, word_index: int) -> str: ...
    def get_src_word_index(self, word: str) -> int: ...
    def src_word_exists(self, word: str) -> bool: ...

    def add_trg_word(self, word: str) -> int: ...
    def get_trg_word(self, word_index: int) -> str: ...
    def get_trg_word_index(self, word: str) -> int: ...
    def trg_word_exists(self, word: str) -> bool: ...

    def start_training(self) -> None: ...
    def train(self) -> None: ...
    def end_training(self) -> None: ...

    def get_sentence_length_log_prob(self, src_length: int, trg_length: int) -> float: ...
    def get_sentence_length_prob(self, src_length: int, trg_length: int) -> float: ...

    def get_translation_log_prob(self, src_word_index: int, trg_word_index: int) -> float: ...
    def get_translation_prob(self, src_word_index: int, trg_word_index: int) -> float: ...

    def get_translations(self, s: int, threshold: float = 0) -> Sequence[Tuple[int, float]]: ...
    
    @overload
    def get_best_alignment(self, src_sentence: str, trg_sentence: str) -> Tuple[float, Sequence[int]]: ...
    @overload
    def get_best_alignment(self, src_sentence: Sequence[int], trg_sentence: Sequence[int]) -> Tuple[float, Sequence[int]]: ...
    @overload
    def get_best_alignment(self, src_sentence: Sequence[str], trg_sentence: Sequence[str]) -> Tuple[float, Sequence[int]]: ...

    @overload
    def get_best_alignments(self, src_sentences: Sequence[Sequence[int]], trg_sentences: Sequence[Sequence[int]]) -> Sequence[Tuple[float, Sequence[int]]]: ...
    @overload
    def get_best_alignments(self, src_sentences: Sequence[Sequence[str]], trg_sentences: Sequence[Sequence[str]]) -> Sequence[Tuple[float, Sequence[int]]]: ...

    def load(self, prefix_filename: str) -> None: ...
    def print(self, prefix_filename: str) -> None: ...

    def clear(self) -> None: ...


class IncrAlignmentModel(AlignmentModel):
    def start_incr_training(self, sentence_pair_range: Tuple[int, int]) -> None: ...
    def incr_train(self, sentence_pair_range: Tuple[int, int]) -> None: ...
    def end_incr_training(self) -> None: ...


class Ibm1AlignmentModel(AlignmentModel):
    def __init__(self) -> None: ...


class IncrIbm1AlignmentModel(Ibm1AlignmentModel, IncrAlignmentModel):
    def __init__(self) -> None: ...


class Ibm2AlignmentModel(Ibm1AlignmentModel, AlignmentModel):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, model: Ibm1AlignmentModel) -> None: ...

    def get_alignment_log_prob(self, j: int, src_length: int, trg_length: int, i: int) -> float: ...
    def get_alignment_prob(self, j: int, src_length: int, trg_length: int, i: int) -> float: ...


class IncrIbm2AlignmentModel(Ibm2AlignmentModel, IncrAlignmentModel):
    def __init__(self) -> None: ...


class HmmAlignmentModel(Ibm1AlignmentModel, AlignmentModel):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, model: Ibm1AlignmentModel) -> None: ...

    def get_alignment_log_prob(self, prev_i: int, src_length: int, i: int) -> float: ...
    def get_alignment_prob(self, prev_i: int, src_length: int, i: int) -> float: ...


class IncrHmmAlignmentModel(HmmAlignmentModel, IncrAlignmentModel):
    def __init__(self) -> None: ...


class FastAlignModel(IncrAlignmentModel):
    def __init__(self) -> None: ...

    def get_alignment_log_prob(self, j: int, src_length: int, trg_length: int, i: int) -> float: ...
    def get_alignment_prob(self, j: int, src_length: int, trg_length: int, i: int) -> float: ...


class Ibm3AlignmentModel(Ibm2AlignmentModel, Ibm1AlignmentModel, AlignmentModel):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, model: HmmAlignmentModel) -> None: ...
    @overload
    def __init__(self, model: Ibm2AlignmentModel) -> None: ...

    def get_distortion_log_prob(self, i: int, src_length: int, trg_length: int, j: int) -> float: ...
    def get_distortion_prob(self, i: int, src_length: int, trg_length: int, j: int) -> float: ...

    def get_fertility_log_prob(self, src_word_index: int, fertility: int) -> float: ...
    def get_fertility_prob(self, src_word_index: int, fertility: int) -> float: ...


class Ibm4AlignmentModel(Ibm3AlignmentModel, Ibm2AlignmentModel, Ibm1AlignmentModel, AlignmentModel):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, model: Ibm3AlignmentModel) -> None: ...

    def get_head_distortion_log_prob(self, src_word_class: int, trg_word_class: int, trg_length: int, dj: int) -> float: ...
    def get_head_distortion_prob(self, src_word_class: int, trg_word_class: int, trg_length: int, dj: int) -> float: ...

    def get_nonhead_distortion_log_prob(self, trg_word_class: int, trg_length: int, dj: int) -> float: ...
    def get_nonhead_distortion_prob(self, trg_word_class: int, trg_length: int, dj: int) -> float: ...


__all__ = [
    "AlignmentModel",
    "FastAlignModel",
    "HmmAlignmentModel",
    "Ibm1AlignmentModel",
    "Ibm2AlignmentModel",
    "Ibm3AlignmentModel",
    "Ibm4AlignmentModel",
    "IncrAlignmentModel",
    "IncrHmmAlignmentModel",
    "IncrIbm1AlignmentModel",
    "IncrIbm2AlignmentModel"
]
