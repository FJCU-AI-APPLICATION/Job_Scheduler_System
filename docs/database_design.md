# 1. **Introduction**

This database design aims to address the following core requirements:

1. **Employee Management**: Store employee details (e.g., name, phone, type).  
2. **Unavailability Tracking**: Record which days or date ranges each employee is unavailable.  
3. **AI Model Management**: Keep track of AI models (their versions, file paths, etc.).  
4. **Shift Policy Definition**: Define rules on how many shifts exist per day, start/end times, and which AI model is used to optimize those shifts.  
5. **Daily Scheduling**: Apply policies to create daily schedules, optionally assigning employees to specific shifts.

The design is intended for **Microsoft SQL Server** (or a similar RDBMS). However, it can be adapted to other SQL dialects by adjusting data types and functions (e.g., `GETDATE()`).

---

# 2. **Key Entities and Tables**

1. **Employee**  
   - Stores an individual worker’s info (name, contact, etc.).  
2. **EmployeeUnavailability**  
   - Tracks which days or date ranges an employee is **not** available to work.  
3. **AiModel**  
   - Stores metadata for each AI model used to optimize schedules.  
4. **ShiftPolicy**  
   - High-level template describing a set of shifts (morning, evening, etc.). Binds **exactly one** AI model.  
5. **ShiftPolicyDetail**  
   - For each shift policy, stores start/end times for **each** shift in the policy.  
6. **Schedule** (optional in some designs)  
   - Represents daily or weekly instances of a shift policy. Ties a specific date (or date range) to a policy.  
7. **ScheduleEmployee** (optional “bridge” table)  
   - Bridging table to link schedules and employees.

---

# 3. **Table Definitions**

Below is a high-level SQL DDL outline for each table. Adjust the schema (datatypes, constraints, naming) to match your environment and internal conventions.

---

## 3.1 **Employee**

```sql
CREATE TABLE Employee (
    id BIGINT IDENTITY(1,1) NOT NULL,
    name NVARCHAR(100) NOT NULL,
    age INT NOT NULL CHECK (age >= 0),
    phone NVARCHAR(20) NOT NULL,
    identity NVARCHAR(4) NOT NULL,     -- e.g. 'FULL' (full-time), 'PART' (part-time)
    salary_type NVARCHAR(5) NOT NULL,  -- e.g. 'MONTH', 'HOUR'
    insert_date DATE NOT NULL DEFAULT GETDATE(),
    update_date DATE NOT NULL DEFAULT GETDATE(),
    PRIMARY KEY (id)
);
```

### **Purpose**  

- Stores basic data about each employee.  
- `id` is `BIGINT` to allow a large ID range.

### **Constraints**  

- **`CHECK (age >= 0)`** ensures no negative ages.  
- **`PRIMARY KEY`** on `id`.

---

## 3.2 **EmployeeUnavailability**

We combine **day-of-week** and **date-range** unavailability into one table using a **type field**:

```sql
CREATE TABLE EmployeeUnavailability (
    id BIGINT IDENTITY(1,1) NOT NULL,
    employee_id BIGINT NOT NULL,
    
    unavailability_type VARCHAR(20) NOT NULL, 
    -- e.g. 'DAY_OF_WEEK' or 'DATE_RANGE'
    
    day_of_week TINYINT NULL, 
    -- 1=Mon, 2=Tue, etc. Only valid if unavailability_type='DAY_OF_WEEK'.
    
    start_date DATE NULL,
    end_date DATE NULL,
    -- For date-based unavailability if unavailability_type='DATE_RANGE'.
    
    reason NVARCHAR(255),
    PRIMARY KEY (id),

    CONSTRAINT FK_EmployeeUnavailability_Employee
        FOREIGN KEY (employee_id) REFERENCES Employee(id)
);
```

### **Purpose**  

- Tracks unavailability either by day-of-week (e.g., every Friday) or by date range (e.g., vacation from 2024-03-10 to 2024-03-15).

### **Constraints**  

- **`FOREIGN KEY (employee_id)`** references **Employee(id)**.  
- **Optional**: A **CHECK** constraint to enforce that if `unavailability_type='DAY_OF_WEEK'`, then `day_of_week` is NOT NULL and `start_date/end_date` are NULL, and vice versa for `DATE_RANGE`.

---

## 3.3 **AiModel**

```sql
CREATE TABLE AiModel (
    id BIGINT IDENTITY(1,1) NOT NULL,
    model_name NVARCHAR(100) NOT NULL,   -- e.g. "GPT-based scheduler"
    model_version NVARCHAR(50) NOT NULL, -- e.g. "v1.2"
    model_path NVARCHAR(255),            -- path or URL of the model
    description NVARCHAR(MAX),
    PRIMARY KEY (id)
);
```

### **Purpose**  

- Stores metadata for AI models used to optimize or generate schedules.

### **Constraints**  

- **`PRIMARY KEY (id)`**.

---

## 3.4 **ShiftPolicy**

```sql
CREATE TABLE ShiftPolicy (
    id BIGINT IDENTITY(1,1) NOT NULL,
    policy_name NVARCHAR(100) NOT NULL,   -- e.g. "Standard 2 Shifts"
    description NVARCHAR(MAX),
    
    ai_model_id BIGINT NOT NULL,          -- link to AiModel

    PRIMARY KEY (id),
    CONSTRAINT FK_ShiftPolicy_AiModel
        FOREIGN KEY (ai_model_id) REFERENCES AiModel(id)
);
```

### **Purpose**  

- Defines a high-level scheduling strategy (e.g., 2-shifts per day).  
- Binds to exactly one AI model for schedule optimization.

### **Constraints**  

- **`FOREIGN KEY (ai_model_id)`** references **AiModel(id)**.

---

## 3.5 **ShiftPolicyDetail**

```sql
CREATE TABLE ShiftPolicyDetail (
    id BIGINT IDENTITY(1,1) NOT NULL,
    policy_id BIGINT NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT FK_ShiftPolicyDetail_Policy
        FOREIGN KEY (policy_id) REFERENCES ShiftPolicy(id)
);
```

### **Purpose**  

- Lists each shift (start/end time) that belongs to a specific `ShiftPolicy`.  
- For example, shift #1 = 08:00–16:00, shift #2 = 16:00–00:00.

### **Constraints**  

- **`FOREIGN KEY (policy_id)`** references **ShiftPolicy(id)**.

---

## 3.6 **Schedule** (Optional)

If you need **daily** or **date-based** schedules referencing a policy:

```sql
CREATE TABLE Schedule (
    id BIGINT IDENTITY(1,1) NOT NULL,
    schedule_date DATE NOT NULL,
    policy_id BIGINT NOT NULL,
    manager_id BIGINT,           -- who manages today's schedule?
    notes NVARCHAR(MAX),
    PRIMARY KEY (id),

    CONSTRAINT FK_Schedule_Policy
        FOREIGN KEY (policy_id) REFERENCES ShiftPolicy(id),

    CONSTRAINT FK_Schedule_Manager
        FOREIGN KEY (manager_id) REFERENCES Employee(id)
);
```

### **Purpose**  

- A **day** or **range** of days can use a particular shift policy.  
- Optionally link an **Employee** as the schedule manager.

### **Constraints**  

- **`FOREIGN KEY (policy_id)`** ensures the schedule uses an existing policy.  
- **`FOREIGN KEY (manager_id)`** references **Employee(id)** if you track manager responsibility.

---

## 3.7 **ScheduleEmployee** (Optional Many-to-Many Bridge)

If you want to assign **multiple employees** to the same schedule or shift, create a bridging table:

```sql
CREATE TABLE ScheduleEmployee (
    id BIGINT IDENTITY(1,1) NOT NULL,
    schedule_id BIGINT NOT NULL,
    employee_id BIGINT NOT NULL,
    assigned_date DATETIME NOT NULL DEFAULT GETDATE(),
    PRIMARY KEY (id),

    CONSTRAINT FK_ScheduleEmployee_Schedule
        FOREIGN KEY (schedule_id) REFERENCES Schedule(id),

    CONSTRAINT FK_ScheduleEmployee_Employee
        FOREIGN KEY (employee_id) REFERENCES Employee(id)
);
```

### **Purpose**  

- Ties employees to a schedule.  
- Could also track which shift within the policy they are assigned to if you add a column referencing **`ShiftPolicyDetail`**.

---

# 4. **Entity-Relationship Overview**

Below is a textual “diagram” of how these tables link:

```
 Employee --------------------< EmployeeUnavailability
      |                                  ^
      | (manager_id)                     |
      v                                  |
 Schedule  >------------------ ShiftPolicy  >----------- AiModel
   | (policy_id)                    ^ (policy_id)
   v                                 |
 ScheduleEmployee                    ShiftPolicyDetail
    (employee_id)                    (policy_id)
```

- **`AiModel`** : Central store of AI models.  
- **`ShiftPolicy`** : Each policy references exactly one `AiModel`.  
- **`ShiftPolicyDetail`** : Each policy can have multiple details (shifts).  
- **`Schedule`** : A day or range that references one `ShiftPolicy`. Optionally, a manager (`Employee.id`).  
- **`ScheduleEmployee`** : Connects multiple employees to a schedule.  
- **`EmployeeUnavailability`** : Tracks days or date ranges an employee is unavailable.

---

# 5. **Data Flow Examples**

1. **Define AI Model**  
   - Insert into `AiModel` (e.g., `model_name="ScheduleOptimizer", version="v1.0"`).  

2. **Define a Shift Policy**  
   - Insert into `ShiftPolicy` referencing `AiModel(id)`.  
   - Insert **two** or **three** shifts into `ShiftPolicyDetail` for that policy.  

3. **Create a Schedule** for a specific date  
   - Insert a row in `Schedule` referencing the `ShiftPolicy(id)` and optional manager.  

4. **Assign Employees** to the schedule  
   - Insert rows into `ScheduleEmployee` for each employee working that day.  
   - If you need to store **which shift** from the policy, you can add a `shift_detail_id` column in `ScheduleEmployee` referencing `ShiftPolicyDetail`.

5. **Record Unavailability**  
   - If an employee is always off on Fridays, insert a row in `EmployeeUnavailability` with `unavailability_type='DAY_OF_WEEK'`, `day_of_week=5`.  
   - If an employee is on vacation for a week, insert `start_date` and `end_date`.

---

# 6. **Constraints & Considerations**

1. **Foreign Key Cascade**  
   - By default, we use `FOREIGN KEY … REFERENCES …`. Decide if you need `ON DELETE CASCADE` or `ON DELETE SET NULL`. For example, if you delete an AI model, do you want all policies referencing it also gone, or do you want to forbid deletion?

2. **Data Type Consistency**  
   - We use `BIGINT` for all primary keys to keep them consistent. If you prefer `INT`, ensure **all** referencing columns match.

3. **Time Validations**  
   - If needed, you can add a check constraint to ensure `end_time` is after `start_time`.  

4. **Optional Nested Keys**  
   - For partial day unavailability or advanced scheduling scenarios, you could add more columns (e.g., `start_time`, `end_time` in `EmployeeUnavailability`).

5. **Performance Indexes**  
   - Create indexes for queries you expect frequently: e.g., `(employee_id)` on `EmployeeUnavailability`, `(policy_id)` on `ShiftPolicyDetail`, etc.
