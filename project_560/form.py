from flask_wtf import FlaskForm
from wtforms import StringField, Submitfield
from wtfforms.validators import DataRequired, Length

class RegistrationForm(FlaskForm):
    stock_code = StringField('Stock_code', validators=[DataRequired(), Length(min=2, max=8)])
    submit = Submitfield('Submit')
