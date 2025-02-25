## Deployment Guide

To deploy the backend , follow the steps below:

1. Clone the repository:

```bash
git clone git@github.com:danghoangnhan/Job_Scheduler_System.git
cd Job_Scheduler_System/server
```

2. Create a prod.env file in the envs directory. Update the environment variables with your desired configuration:

```plaintext
# envs/dev.env file

# Database configuration
DB_NAME=db-name
DB_USER=db-user
DB_PASSWORD=@ai113
DB_HOST=db-host
DB_PORT=
```

1. Start the services (Django application):

```bash
    python manage.py runserver
```

4. Once the services are up and running, open your web browser and go to `http://127.0.0.1:8000/admin/` to access the Django Admin interface.
