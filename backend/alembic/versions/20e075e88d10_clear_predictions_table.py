"""Clear predictions table

Revision ID: 20e075e88d10
Revises: 266ac36ac10f
Create Date: 2025-08-23 02:13:02.465662

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20e075e88d10'
down_revision: Union[str, Sequence[str], None] = '266ac36ac10f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.execute("DELETE FROM predictions")

def downgrade():
    pass
