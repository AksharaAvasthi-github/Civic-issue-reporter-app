from alembic import op
import sqlalchemy as sa

revision = 'ef705f485284'
down_revision = '7c1c58ac5e28'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Remove SQLite incompatible commands
    # op.alter_column('users', 'username',
    #            existing_type=sa.VARCHAR(),
    #            nullable=False)
    # op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)

    # Add 'reported_by' column to 'predictions' table
    op.add_column('predictions', sa.Column('reported_by', sa.Integer(), nullable=True))

    # Add ForeignKey constraint separately (because SQLite doesn't support inline FK in add_column)
    op.create_foreign_key(
        'fk_predictions_reported_by_users',
        'predictions', 'users',
        ['reported_by'], ['id']
    )


def downgrade() -> None:
    # Remove ForeignKey and column
    op.drop_constraint('fk_predictions_reported_by_users', 'predictions', type_='foreignkey')
    op.drop_column('predictions', 'reported_by')

    # Remove the index if you added it manually (commented here as it was not added)
    # op.drop_index(op.f('ix_users_id'), table_name='users')

    # Don't alter column (SQLite limitation)
    # op.alter_column('users', 'username',
    #            existing_type=sa.VARCHAR(),
    #            nullable=True)
