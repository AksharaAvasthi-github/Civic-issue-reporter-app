"""Add reported_by column to predictions

Revision ID: 7c1c58ac5e28
Revises: 20e075e88d10
Create Date: 2025-08-24 01:43:09.067795

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7c1c58ac5e28'
down_revision: Union[str, Sequence[str], None] = '20e075e88d10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    with op.batch_alter_table("predictions", schema=None) as batch_op:
        batch_op.add_column(sa.Column('reported_by', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            'fk_reported_by_users', 'users', ['reported_by'], ['id']
        )

def downgrade():
    with op.batch_alter_table("predictions", schema=None) as batch_op:
        batch_op.drop_constraint('fk_reported_by_users', type_='foreignkey')
        batch_op.drop_column('reported_by')
