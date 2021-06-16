from django.urls import path
from if_gui import views

urlpatterns = [
    path("", views.home, name="home"),
    path("if_gui/<name>", views.hello_there, name="hello_there"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
]