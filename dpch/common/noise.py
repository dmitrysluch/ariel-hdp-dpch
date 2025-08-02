from abc import ABC, abstractmethod

import numpy as np

from dpch.common.queries.interface import DPQueryMixin
from dpch.common.schema import SchemaDataset
from dpch.common.session import Session

# Please refactor noise interface and implementations once again.
# Let noise constructor receive just a list of scaling parameters
# (for each column) and apply method just a numpy array
# of corresponding shape. But there exists a static factory method
# which recieves query, session, schema_dataset and returns updated session
# and instantiated noise.
#
# Also please make a wrapper which checks session type and selects corresponding noise


class NoiseMixin(ABC):
    def __init__(self, scales: list[float]):
        """
        Initialize noise with scaling parameters for each column
        Args:
            scales: List of scaling parameters, one for each column
        """
        self.scales = scales

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply noise to the input array
        Args:
            data: Clean numpy array to add noise to
        Returns:
            np.ndarray: Noised data
        """
        pass

    @staticmethod
    @abstractmethod
    def from_query(
        query: DPQueryMixin,
        session: Session,
        schema_dataset: SchemaDataset,
    ) -> tuple[Session, "NoiseMixin"]:
        """
        Factory method to create noise instance from query parameters
        Args:
            query: Query object for sensitivity calculation
            session: Current session with privacy budget
            schema_dataset: Schema dataset for query validation
        Returns:
            tuple[Session, NoiseMixin]: Updated session and noise instance
        """
        pass


class NotEnoughPrivacyBudget(Exception):
    pass


class LaplacianNoise(NoiseMixin):
    def __init__(self, scales: list[float]):
        super().__init__(scales)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply Laplacian noise column-wise"""
        if data.shape[1] != len(self.scales):
            raise ValueError(
                f"Data has {data.shape[1]} columns but {len(self.scales)} scales provided"
            )

        noised_data = np.empty_like(data)
        for col_idx in range(data.shape[1]):
            noised_data[:, col_idx] = data[:, col_idx] + np.random.laplace(
                loc=0, scale=self.scales[col_idx], size=data.shape[0]
            )
        return noised_data

    @staticmethod
    def from_query(
        query: DPQueryMixin,
        session: Session,
        schema_dataset: SchemaDataset,
    ) -> tuple[Session, "LaplacianNoise"]:
        n_rows, n_cols = query.shape(schema_dataset)

        # Calculate epsilon per cell
        eps_per_cell = session.max_eps / session.queries_total
        # Calculate epsilon per column
        eps_per_col = eps_per_cell * n_rows
        # Calculate total epsilon used
        total_epsilon = eps_per_col * n_cols

        # Calculate sensitivity for each column
        sensitivities = query.sensitivity_over_columns(schema_dataset, "l1")

        # Calculate scales for each column
        scales = [sensitivity / eps_per_col for sensitivity in sensitivities]

        # Update session
        updated_session = session.model_copy()
        updated_session.eps_used += total_epsilon
        updated_session.queries_left -= n_rows * n_cols

        if (
            updated_session.eps_used > updated_session.max_eps
            or updated_session.queries_left < 0
        ):
            raise NotEnoughPrivacyBudget("Not enough privacy budget for query")

        return updated_session, LaplacianNoise(scales)


class GaussianNoise(NoiseMixin):
    def __init__(self, scales: list[float]):
        super().__init__(scales)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise column-wise"""
        if data.shape[1] != len(self.scales):
            raise ValueError(
                f"Data has {data.shape[1]} columns but {len(self.scales)} scales provided"
            )

        noised_data = np.empty_like(data)
        for col_idx in range(data.shape[1]):
            noised_data[:, col_idx] = data[:, col_idx] + np.random.normal(
                loc=0, scale=self.scales[col_idx], size=data.shape[0]
            )
        return noised_data

    @staticmethod
    def from_query(
        query: DPQueryMixin,
        session: Session,
        schema_dataset: SchemaDataset,
    ) -> tuple[Session, "GaussianNoise"]:
        n_rows, n_cols = query.shape(schema_dataset)

        # Calculate epsilon per cell using advanced composition
        eps_per_cell = 1 / np.sqrt(session.queries_total)
        # Calculate epsilon per column using advanced composition
        eps_per_col = eps_per_cell * np.sqrt(n_rows)
        # Calculate total epsilon used
        total_epsilon = eps_per_col * np.sqrt(n_cols)

        # Calculate delta per cell
        delta_per_cell = session.max_delta / session.queries_total
        # Calculate delta per column
        delta_per_col = delta_per_cell * n_rows
        # Calculate total delta used
        total_delta = delta_per_col * n_cols

        # Calculate sensitivity for each column
        sensitivities = query.sensitivity_over_columns(schema_dataset, "l2")

        # Calculate scales for each column (standard deviation for Gaussian mechanism)
        scales = [
            sensitivity * np.sqrt(2 * np.log(1.25 / delta_per_col)) / eps_per_col
            for sensitivity in sensitivities
        ]

        # Update session
        updated_session = session.model_copy()
        updated_session.eps_used += total_epsilon
        updated_session.delta_used += total_delta
        updated_session.queries_left -= n_rows * n_cols

        if (
            updated_session.eps_used > updated_session.max_eps
            or updated_session.delta_used > updated_session.max_delta
            or updated_session.queries_left < 0
        ):
            raise NotEnoughPrivacyBudget("Not enough privacy budget for query")

        return updated_session, GaussianNoise(scales)


def create_noise_from_query(
    query: DPQueryMixin,
    session: Session,
    schema_dataset: SchemaDataset,
) -> tuple[Session, NoiseMixin]:
    """
    Factory function that creates appropriate noise based on session noise_type
    Args:
        query: Query object for sensitivity calculation
        session: Current session with privacy budget and noise type
        schema_dataset: Schema dataset for query validation
    Returns:
        tuple[Session, NoiseMixin]: Updated session and noise instance
    """
    if session.noise_type == "laplacian":
        return LaplacianNoise.from_query(query, session, schema_dataset)
    elif session.noise_type == "gaussian":
        return GaussianNoise.from_query(query, session, schema_dataset)
    else:
        raise ValueError(f"Unsupported noise type: {session.noise_type}")
