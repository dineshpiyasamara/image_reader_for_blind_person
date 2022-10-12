from django.db import models


class Upload(models.Model):
    image = models.ImageField(upload_to="images/")
