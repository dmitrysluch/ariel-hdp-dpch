from abc import ABC, abstractmethod

import numpy as np

from dpch.common.queries.interface import DPQueryMixin
from dpch.common.schema import SchemaDataset
from dpch.common.session import Session


# Please fix NoiseMixin. Now evaluate must accept clean dataframe, query, session, schema dataset. Noised data and updated session is returned
class NoiseMixin(ABC):
    @abstractmethod
    def evaluate(
        self,
        data: np.ndarray,
        session: Session,
    ) -> tuple[np.ndarray, Session]:
        """
        Add noise to the input array and update session
        Args:
            data: Clean dataframe to add noise to
            query: Query object for sensitivity calculation
            session: Current session with privacy budget
            schema_dataset: Schema dataset for query validation
        Returns:
            tuple[np.ndarray, Session]: Noised data and updated session
        """
        pass


class NotEnoughPrivacyBudget(Exception):
    pass


# Please rewrite Laplacian noise to calculate epsilon on fly from session.
# Each cell in resulting dataframe is considered as a query.
# First eps is calculated for a cell, the eps for column
# is considered to be eps for a cell multiplied by dataset length. The scale
# for a column is column sensitivity divided by column epsilon.
# Total epsilon is column epsilon multiplied on number of columns.
class LaplacianNoise(NoiseMixin):
    def __init__(
        self,
        query: DPQueryMixin,
        session: Session,
        schema_dataset: SchemaDataset,
        eps_per_cell_override=None,
    ):
        self.n_rows, self.n_cols = query.shape(schema_dataset)
        # Calculate epsilon per cell
        eps_per_cell = (
            session.max_eps / session.queries_total
            if eps_per_cell_override is None
            else eps_per_cell_override
        )
        # Calculate epsilon per column
        self.eps_per_col = eps_per_cell * self.n_rows
        # Calculate total epsilon used
        self.total_epsilon = self.eps_per_col * self.n_cols
        # Calculate sensitivity for each column
        self.sensitivities = query.sensitivity_over_columns(schema_dataset, "l1")

    def evaluate(
        self,
        data: np.ndarray,
        session: Session,
    ) -> tuple[np.ndarray, Session]:
        # Add Laplacian noise column-wise
        noised_data = np.empty_like(data)
        for col_idx in range(self.n_cols):
            scale = self.sensitivities[col_idx] / self.eps_per_col
            noised_data[:, col_idx] = data[:, col_idx] + np.random.laplace(
                loc=0, scale=scale, size=self.n_rows
            )
        # Update session
        updated_session = session.model_copy()
        updated_session.eps_used += self.total_epsilon
        updated_session.queries_left -= self.n_rows * self.n_cols
        if (
            updated_session.eps_used > updated_session.max_eps
            or updated_session.queries_left < 0
        ):
            raise NotEnoughPrivacyBudget("Not enough privacy budget for query")
        return noised_data, updated_session


# Please rewrite gaussian noise in the same manner as laplacian noise. The eps_per_cell is now 1 / sqrt(queries_total), eps_per_column eps_per_cell * sqrt(self.n_rows) (Advanced composition for gaussian noise is used).
class GaussianNoise(NoiseMixin):
    def __init__(
        self,
        query: DPQueryMixin,
        session: Session,
        schema_dataset: SchemaDataset,
        eps_per_cell_override=None,
        delta_per_cell_override=None,
    ):
        self.n_rows, self.n_cols = query.shape(schema_dataset)
        # Calculate epsilon per cell using advanced composition
        eps_per_cell = (
            1 / np.sqrt(session.queries_total)
            if eps_per_cell_override is None
            else eps_per_cell_override
        )
        # Calculate epsilon per column using advanced composition
        self.eps_per_col = eps_per_cell * np.sqrt(self.n_rows)
        # Calculate total epsilon used
        self.total_epsilon = self.eps_per_col * np.sqrt(self.n_cols)

        # Calculate delta per cell
        delta_per_cell = (
            session.max_delta / session.queries_total
            if delta_per_cell_override is None
            else delta_per_cell_override
        )
        # Calculate delta per column
        self.delta_per_col = delta_per_cell * self.n_rows
        # Calculate total delta used
        self.total_delta = self.delta_per_col * self.n_cols

        # Calculate sensitivity for each column
        self.sensitivities = query.sensitivity_over_columns(schema_dataset, "l2")

    def evaluate(
        self,
        data: np.ndarray,
        session: Session,
    ) -> tuple[np.ndarray, Session]:
        # Add Gaussian noise column-wise
        noised_data = np.empty_like(data)
        for col_idx in range(self.n_cols):
            # Calculate noise standard deviation for Gaussian mechanism
            std = (
                self.sensitivities[col_idx]
                * np.sqrt(2 * np.log(1.25 / self.delta_per_col))
                / self.eps_per_col
            )
            noised_data[:, col_idx] = data[:, col_idx] + np.random.normal(
                loc=0, scale=std, size=self.n_rows
            )

        # Update session (deduct epsilon and delta from budget)
        updated_session = session.model_copy()
        updated_session.eps_used += self.total_epsilon
        updated_session.delta_used += self.total_delta
        updated_session.queries_left -= self.n_rows * self.n_cols

        if (
            updated_session.eps_used > updated_session.max_eps
            or updated_session.queries_left < 0
        ):
            raise NotEnoughPrivacyBudget("Not enough privacy budget for query")
        return noised_data, updated_session
