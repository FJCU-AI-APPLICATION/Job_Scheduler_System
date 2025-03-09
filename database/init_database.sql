-- Database Client 8.1.9
 -- Host: 211.20.21.35 Port: 9493 Database: dbo  Schema: dbo 

DROP TABLE IF EXISTS "AiModel";
CREATE TABLE "AiModel"(
    id int IDENTITY(1,1) NOT NULL,
    model_name nvarchar(100) NOT NULL,
    model_version nvarchar(50) NOT NULL,
    model_path nvarchar(255),
    description nvarchar(max),
    PRIMARY KEY(id)
);

DROP TABLE IF EXISTS auth_group;
CREATE TABLE auth_group(
    id int IDENTITY(1,1) NOT NULL,
    name nvarchar(150) NOT NULL,
    PRIMARY KEY(id)
);
CREATE UNIQUE INDEX auth_group_name_a6ea08ec_uniq ON auth_group("name");

DROP TABLE IF EXISTS auth_group_permissions;
CREATE TABLE auth_group_permissions(
    id int IDENTITY(1,1) NOT NULL,
    group_id int NOT NULL,
    permission_id int NOT NULL,
    PRIMARY KEY(id),
    CONSTRAINT auth_group_permissions_permission_id_84c5c92e_fk_auth_permission_id FOREIGN key(permission_id) REFERENCES auth_permission(id),
    CONSTRAINT auth_group_permissions_group_id_b120cbf9_fk_auth_group_id FOREIGN key(group_id) REFERENCES auth_group(id)
);
CREATE INDEX auth_group_permissions_permission_id_84c5c92e ON auth_group_permissions("permission_id");
CREATE UNIQUE INDEX auth_group_permissions_group_id_permission_id_0cd325b0_uniq ON auth_group_permissions("group_id","permission_id");
CREATE INDEX auth_group_permissions_group_id_b120cbf9 ON auth_group_permissions("group_id");

DROP TABLE IF EXISTS auth_permission;
CREATE TABLE auth_permission(
    id int IDENTITY(1,1) NOT NULL,
    name nvarchar(255) NOT NULL,
    content_type_id int NOT NULL,
    codename nvarchar(100) NOT NULL,
    PRIMARY KEY(id),
    CONSTRAINT auth_permission_content_type_id_2f476e4b_fk_django_content_type_id FOREIGN key(content_type_id) REFERENCES django_content_type(id)
);
CREATE INDEX auth_permission_content_type_id_2f476e4b ON auth_permission("content_type_id");
CREATE UNIQUE INDEX auth_permission_content_type_id_codename_01ab375a_uniq ON auth_permission("content_type_id","codename");

DROP TABLE IF EXISTS auth_user;
CREATE TABLE auth_user(
    id int IDENTITY(1,1) NOT NULL,
    password nvarchar(128) NOT NULL,
    last_login datetimeoffset,
    is_superuser bit NOT NULL,
    username nvarchar(150) NOT NULL,
    first_name nvarchar(150) NOT NULL,
    last_name nvarchar(150) NOT NULL,
    email nvarchar(254) NOT NULL,
    is_staff bit NOT NULL,
    is_active bit NOT NULL,
    date_joined datetimeoffset NOT NULL,
    PRIMARY KEY(id)
);
CREATE UNIQUE INDEX auth_user_username_6821ab7c_uniq ON auth_user("username");

DROP TABLE IF EXISTS auth_user_groups;
CREATE TABLE auth_user_groups(
    id int IDENTITY(1,1) NOT NULL,
    user_id int NOT NULL,
    group_id int NOT NULL,
    PRIMARY KEY(id),
    CONSTRAINT auth_user_groups_group_id_97559544_fk_auth_group_id FOREIGN key(group_id) REFERENCES auth_group(id),
    CONSTRAINT auth_user_groups_user_id_6a12ed8b_fk_auth_user_id FOREIGN key(user_id) REFERENCES auth_user(id)
);
CREATE INDEX auth_user_groups_user_id_6a12ed8b ON auth_user_groups("user_id");
CREATE INDEX auth_user_groups_group_id_97559544 ON auth_user_groups("group_id");
CREATE UNIQUE INDEX auth_user_groups_user_id_group_id_94350c0c_uniq ON auth_user_groups("user_id","group_id");

DROP TABLE IF EXISTS auth_user_user_permissions;
CREATE TABLE auth_user_user_permissions(
    id int IDENTITY(1,1) NOT NULL,
    user_id int NOT NULL,
    permission_id int NOT NULL,
    PRIMARY KEY(id),
    CONSTRAINT auth_user_user_permissions_permission_id_1fbb5f2c_fk_auth_permission_id FOREIGN key(permission_id) REFERENCES auth_permission(id),
    CONSTRAINT auth_user_user_permissions_user_id_a95ead1b_fk_auth_user_id FOREIGN key(user_id) REFERENCES auth_user(id)
);
CREATE UNIQUE INDEX auth_user_user_permissions_user_id_permission_id_14a6b632_uniq ON auth_user_user_permissions("user_id","permission_id");
CREATE INDEX auth_user_user_permissions_permission_id_1fbb5f2c ON auth_user_user_permissions("permission_id");
CREATE INDEX auth_user_user_permissions_user_id_a95ead1b ON auth_user_user_permissions("user_id");

DROP TABLE IF EXISTS "Calendar";
CREATE TABLE "Calendar"(
    年月份 char(6) NOT NULL,
    "F01" nvarchar(50),
    "F02" nvarchar(50),
    "F03" nvarchar(50),
    "F04" nvarchar(50),
    "F05" nvarchar(50),
    "F06" nvarchar(50),
    "F07" nvarchar(50),
    "F08" nvarchar(50),
    "F09" nvarchar(50),
    "F10" nvarchar(50),
    "F11" nvarchar(50),
    "F12" nvarchar(50),
    "F13" nvarchar(50),
    "F14" nvarchar(50),
    "F15" nvarchar(50),
    "F16" nvarchar(50),
    "F17" nvarchar(50),
    "F18" nvarchar(50),
    "F19" nvarchar(50),
    "F20" nvarchar(50),
    "F21" nvarchar(50),
    "F22" nvarchar(50),
    "F23" nvarchar(50),
    "F24" nvarchar(50),
    "F25" nvarchar(50),
    "F26" nvarchar(50),
    "F27" nvarchar(50),
    "F28" nvarchar(50),
    "F29" nvarchar(50),
    "F30" nvarchar(50),
    "F31" nvarchar(50),
    PRIMARY KEY(年月份)
);

DROP TABLE IF EXISTS django_admin_log;
CREATE TABLE django_admin_log(
    id int IDENTITY(1,1) NOT NULL,
    action_time datetimeoffset NOT NULL,
    object_id nvarchar(max),
    object_repr nvarchar(200) NOT NULL,
    action_flag smallint NOT NULL,
    change_message nvarchar(max) NOT NULL,
    content_type_id int,
    user_id int NOT NULL,
    PRIMARY KEY(id),
    CONSTRAINT django_admin_log_content_type_id_c4bce8eb_fk_django_content_type_id FOREIGN key(content_type_id) REFERENCES django_content_type(id),
    CONSTRAINT django_admin_log_user_id_c564eba6_fk_auth_user_id FOREIGN key(user_id) REFERENCES auth_user(id),
    CONSTRAINT django_admin_log_action_flag_a8637d59_check CHECK ([action_flag]>=(0))
);
CREATE INDEX django_admin_log_content_type_id_c4bce8eb ON django_admin_log("content_type_id");
CREATE INDEX django_admin_log_user_id_c564eba6 ON django_admin_log("user_id");

DROP TABLE IF EXISTS django_content_type;
CREATE TABLE django_content_type(
    id int IDENTITY(1,1) NOT NULL,
    app_label nvarchar(100) NOT NULL,
    model nvarchar(100) NOT NULL,
    PRIMARY KEY(id)
);
CREATE UNIQUE INDEX django_content_type_app_label_model_76bd3d3b_uniq ON django_content_type("app_label","model");

DROP TABLE IF EXISTS django_migrations;
CREATE TABLE django_migrations(
    id int IDENTITY(1,1) NOT NULL,
    app nvarchar(255) NOT NULL,
    name nvarchar(255) NOT NULL,
    applied datetimeoffset NOT NULL,
    PRIMARY KEY(id)
);

DROP TABLE IF EXISTS django_session;
CREATE TABLE django_session(
    session_key nvarchar(40) NOT NULL,
    session_data nvarchar(max) NOT NULL,
    expire_date datetimeoffset NOT NULL,
    PRIMARY KEY(session_key)
);
CREATE INDEX django_session_expire_date_a5c62663 ON django_session("expire_date");

DROP TABLE IF EXISTS "Employee";
CREATE TABLE "Employee"(
    id int IDENTITY(1,1) NOT NULL,
    name nvarchar(100) NOT NULL,
    age int NOT NULL,
    phone nvarchar(20) NOT NULL,
    identity nvarchar(4) NOT NULL,
    salary_type nvarchar(5) NOT NULL,
    insert_date date,
    update_date date,
    PRIMARY KEY(id),
    CONSTRAINT Employee_age_9c1795a5_check CHECK ([age]>=(0))
);

DROP TABLE IF EXISTS "EmployeeUnavailability";
CREATE TABLE "EmployeeUnavailability"(
    id int IDENTITY(1,1) NOT NULL,
    employee_id bigint NOT NULL,
    unavailability_type varchar(20) NOT NULL,
    day_of_week tinyint,
    start_date date,
    end_date date,
    reason nvarchar(255),
    PRIMARY KEY(id),
    CONSTRAINT FK__EmployeeU__emplo__0A9D95DB FOREIGN key(employee_id) REFERENCES "Employee"(id),
    CONSTRAINT CK_EmployeeUnavailability_Type CHECK ([unavailability_type]='DAY_OF_WEEK' AND [day_of_week] IS NOT NULL AND [start_date] IS NULL AND [end_date] IS NULL OR [unavailability_type]='DATE_RANGE' AND [day_of_week] IS NULL AND [start_date] IS NOT NULL AND [end_date] IS NOT NULL)
);

DROP TABLE IF EXISTS "Schedule";
CREATE TABLE "Schedule"(
    id int IDENTITY(1,1) NOT NULL,
    name nvarchar(200) NOT NULL,
    description nvarchar(max) NOT NULL,
    start_date date NOT NULL,
    manager_id bigint,
    PRIMARY KEY(id),
    CONSTRAINT Schedule_manager_id_9c61c003_fk_Employee_id FOREIGN key(manager_id) REFERENCES "Employee"(id)
);

DROP TABLE IF EXISTS "ScheduleEmployee";
CREATE TABLE "ScheduleEmployee"(
    employee_id bigint NOT NULL,
    schedule_id bigint NOT NULL,
    assigned_date datetime NOT NULL DEFAULT (getdate()),
    CONSTRAINT FK_ScheduleEmployee_Employee FOREIGN key(employee_id) REFERENCES "Employee"(id),
    CONSTRAINT FK_ScheduleEmployee_Schedule FOREIGN key(schedule_id) REFERENCES "Schedule"(id)
);

DROP TABLE IF EXISTS "ShiftPolicy";
CREATE TABLE "ShiftPolicy"(
    id int IDENTITY(1,1) NOT NULL,
    policy_name nvarchar(100) NOT NULL,
    description nvarchar(max),
    " ai_model_id" bigint NOT NULL,
    PRIMARY KEY(id),
    CONSTRAINT FK__ShiftPoli__ ai_m__0E6E26BF FOREIGN key(ai_model_id) REFERENCES "AiModel"(id)
);

DROP TABLE IF EXISTS "ShiftPolicyDetail";
CREATE TABLE "ShiftPolicyDetail"(
    id int IDENTITY(1,1) NOT NULL,
    policy_id bigint NOT NULL,
    shift_index int NOT NULL,
    start_time time NOT NULL,
    end_time time NOT NULL,
    PRIMARY KEY(id),
    CONSTRAINT FK__ShiftPoli__polic__07C12930 FOREIGN key(policy_id) REFERENCES "ShiftPolicy"(id)
);

DROP TABLE IF EXISTS sysdiagrams;
CREATE TABLE sysdiagrams(
    name nvarchar(128) NOT NULL,
    principal_id int NOT NULL,
    diagram_id int IDENTITY(1,1) NOT NULL,
    version int,
    definition varbinary(max),
    PRIMARY KEY(diagram_id)
);
CREATE UNIQUE INDEX UK_principal_name ON sysdiagrams("principal_id","name");




INSERT INTO auth_permission(name,content_type_id,codename) VALUES(N'Can add log entry',1,N'add_logentry'),(N'Can change log entry',1,N'change_logentry'),(N'Can delete log entry',1,N'delete_logentry'),(N'Can view log entry',1,N'view_logentry'),(N'Can add permission',2,N'add_permission'),(N'Can change permission',2,N'change_permission'),(N'Can delete permission',2,N'delete_permission'),(N'Can view permission',2,N'view_permission'),(N'Can add group',3,N'add_group'),(N'Can change group',3,N'change_group'),(N'Can delete group',3,N'delete_group'),(N'Can view group',3,N'view_group'),(N'Can add user',4,N'add_user'),(N'Can change user',4,N'change_user'),(N'Can delete user',4,N'delete_user'),(N'Can view user',4,N'view_user'),(N'Can add content type',5,N'add_contenttype'),(N'Can change content type',5,N'change_contenttype'),(N'Can delete content type',5,N'delete_contenttype'),(N'Can view content type',5,N'view_contenttype'),(N'Can add session',6,N'add_session'),(N'Can change session',6,N'change_session'),(N'Can delete session',6,N'delete_session'),(N'Can view session',6,N'view_session');






INSERT INTO django_content_type(app_label,model) VALUES(N'admin',N'logentry'),(N'auth',N'group'),(N'auth',N'permission'),(N'auth',N'user'),(N'contenttypes',N'contenttype'),(N'sessions',N'session');

INSERT INTO django_migrations(app,name,applied) VALUES(N'contenttypes',N'0001_initial','2025-03-02 14:55:37'),(N'auth',N'0001_initial','2025-03-02 14:55:38'),(N'admin',N'0001_initial','2025-03-02 14:55:39'),(N'admin',N'0002_logentry_remove_auto_add','2025-03-02 14:55:39'),(N'admin',N'0003_logentry_add_action_flag_choices','2025-03-02 14:55:39'),(N'contenttypes',N'0002_remove_content_type_name','2025-03-02 14:55:41'),(N'auth',N'0002_alter_permission_name_max_length','2025-03-02 14:55:41'),(N'auth',N'0003_alter_user_email_max_length','2025-03-02 14:55:41'),(N'auth',N'0004_alter_user_username_opts','2025-03-02 14:55:42'),(N'auth',N'0005_alter_user_last_login_null','2025-03-02 14:55:43'),(N'auth',N'0006_require_contenttypes_0002','2025-03-02 14:55:43'),(N'auth',N'0007_alter_validators_add_error_messages','2025-03-02 14:55:43'),(N'auth',N'0008_alter_user_username_max_length','2025-03-02 14:55:45'),(N'auth',N'0009_alter_user_last_name_max_length','2025-03-02 14:55:45'),(N'auth',N'0010_alter_group_name_max_length','2025-03-02 14:55:46'),(N'auth',N'0011_update_proxy_permissions','2025-03-02 14:55:47'),(N'auth',N'0012_alter_user_first_name_max_length','2025-03-02 14:55:47'),(N'sessions',N'0001_initial','2025-03-02 14:55:48'),(N'employee',N'0001_initial','2025-03-02 15:28:58'),(N'schedule',N'0001_initial','2025-03-02 15:28:59');


INSERT INTO "Employee"(name,age,phone,identity,salary_type,insert_date,update_date) VALUES(N'王大明',30,N'123456',N'正職',N'月薪','2024-03-01','2024-03-01'),(N'張小花',26,N'123457',N'正職',N'月薪','2024-03-01','2024-03-01'),(N'李大郎',40,N'234567',N'兼職',N'時薪','2024-03-01','2024-03-01'),(N'李大春',40,N'234569',N'兼職',N'時薪','2024-03-02','2024-03-02'),(N'陳小彬',26,N'123458',N'兼職',N'時薪','2024-03-02','2024-03-02'),(N'Updated Name',35,N'987654321',N'FULL',N'MONTH','2024-03-02','2025-03-03'),(N'韋阿寶',26,N'123466',N'兼職',N'時薪','2024-03-02','2024-03-02'),(N'Updated Name',35,N'987654321',N'FULL',N'MONTH','2025-03-03','2025-03-03');






DROP PROCEDURE IF EXISTS dbo.sp_alterdiagram;

	CREATE PROCEDURE dbo.sp_alterdiagram
	(
		@diagramname 	sysname,
		@owner_id	int	= null,
		@version 	int,
		@definition 	varbinary(max)
	)
	WITH EXECUTE AS 'dbo'
	AS
	BEGIN
		set nocount on
	
		declare @theId 			int
		declare @retval 		int
		declare @IsDbo 			int
		
		declare @UIDFound 		int
		declare @DiagId			int
		declare @ShouldChangeUID	int
	
		if(@diagramname is null)
		begin
			RAISERROR ('Invalid ARG', 16, 1)
			return -1
		end
	
		execute as caller;
		select @theId = DATABASE_PRINCIPAL_ID();	 
		select @IsDbo = IS_MEMBER(N'db_owner'); 
		if(@owner_id is null)
			select @owner_id = @theId;
		revert;
	
		select @ShouldChangeUID = 0
		select @DiagId = diagram_id, @UIDFound = principal_id from dbo.sysdiagrams where principal_id = @owner_id and name = @diagramname 
		
		if(@DiagId IS NULL or (@IsDbo = 0 and @theId <> @UIDFound))
		begin
			RAISERROR ('Diagram does not exist or you do not have permission.', 16, 1);
			return -3
		end
	
		if(@IsDbo <> 0)
		begin
			if(@UIDFound is null or USER_NAME(@UIDFound) is null) -- invalid principal_id
			begin
				select @ShouldChangeUID = 1 ;
			end
		end

		-- update dds data			
		update dbo.sysdiagrams set definition = @definition where diagram_id = @DiagId ;

		-- change owner
		if(@ShouldChangeUID = 1)
			update dbo.sysdiagrams set principal_id = @theId where diagram_id = @DiagId ;

		-- update dds version
		if(@version is not null)
			update dbo.sysdiagrams set version = @version where diagram_id = @DiagId ;

		return 0
	END
	;

DROP PROCEDURE IF EXISTS dbo.sp_creatediagram;

	CREATE PROCEDURE dbo.sp_creatediagram
	(
		@diagramname 	sysname,
		@owner_id		int	= null, 	
		@version 		int,
		@definition 	varbinary(max)
	)
	WITH EXECUTE AS 'dbo'
	AS
	BEGIN
		set nocount on
	
		declare @theId int
		declare @retval int
		declare @IsDbo	int
		declare @userName sysname
		if(@version is null or @diagramname is null)
		begin
			RAISERROR (N'E_INVALIDARG', 16, 1);
			return -1
		end
	
		execute as caller;
		select @theId = DATABASE_PRINCIPAL_ID(); 
		select @IsDbo = IS_MEMBER(N'db_owner');
		revert; 
		
		if @owner_id is null
		begin
			select @owner_id = @theId;
		end
		else
		begin
			if @theId <> @owner_id
			begin
				if @IsDbo = 0
				begin
					RAISERROR (N'E_INVALIDARG', 16, 1);
					return -1
				end
				select @theId = @owner_id
			end
		end
		-- next 2 line only for test, will be removed after define name unique
		if EXISTS(select diagram_id from dbo.sysdiagrams where principal_id = @theId and name = @diagramname)
		begin
			RAISERROR ('The name is already used.', 16, 1);
			return -2
		end
	
		insert into dbo.sysdiagrams(name, principal_id , version, definition)
				VALUES(@diagramname, @theId, @version, @definition) ;
		
		select @retval = @@IDENTITY 
		return @retval
	END
	;

DROP PROCEDURE IF EXISTS dbo.sp_dropdiagram;

	CREATE PROCEDURE dbo.sp_dropdiagram
	(
		@diagramname 	sysname,
		@owner_id	int	= null
	)
	WITH EXECUTE AS 'dbo'
	AS
	BEGIN
		set nocount on
		declare @theId 			int
		declare @IsDbo 			int
		
		declare @UIDFound 		int
		declare @DiagId			int
	
		if(@diagramname is null)
		begin
			RAISERROR ('Invalid value', 16, 1);
			return -1
		end
	
		EXECUTE AS CALLER;
		select @theId = DATABASE_PRINCIPAL_ID();
		select @IsDbo = IS_MEMBER(N'db_owner'); 
		if(@owner_id is null)
			select @owner_id = @theId;
		REVERT; 
		
		select @DiagId = diagram_id, @UIDFound = principal_id from dbo.sysdiagrams where principal_id = @owner_id and name = @diagramname 
		if(@DiagId IS NULL or (@IsDbo = 0 and @UIDFound <> @theId))
		begin
			RAISERROR ('Diagram does not exist or you do not have permission.', 16, 1)
			return -3
		end
	
		delete from dbo.sysdiagrams where diagram_id = @DiagId;
	
		return 0;
	END
	;

DROP PROCEDURE IF EXISTS dbo.sp_helpdiagramdefinition;

	CREATE PROCEDURE dbo.sp_helpdiagramdefinition
	(
		@diagramname 	sysname,
		@owner_id	int	= null 		
	)
	WITH EXECUTE AS N'dbo'
	AS
	BEGIN
		set nocount on

		declare @theId 		int
		declare @IsDbo 		int
		declare @DiagId		int
		declare @UIDFound	int
	
		if(@diagramname is null)
		begin
			RAISERROR (N'E_INVALIDARG', 16, 1);
			return -1
		end
	
		execute as caller;
		select @theId = DATABASE_PRINCIPAL_ID();
		select @IsDbo = IS_MEMBER(N'db_owner');
		if(@owner_id is null)
			select @owner_id = @theId;
		revert; 
	
		select @DiagId = diagram_id, @UIDFound = principal_id from dbo.sysdiagrams where principal_id = @owner_id and name = @diagramname;
		if(@DiagId IS NULL or (@IsDbo = 0 and @UIDFound <> @theId ))
		begin
			RAISERROR ('Diagram does not exist or you do not have permission.', 16, 1);
			return -3
		end

		select version, definition FROM dbo.sysdiagrams where diagram_id = @DiagId ; 
		return 0
	END
	;

DROP PROCEDURE IF EXISTS dbo.sp_helpdiagrams;

	CREATE PROCEDURE dbo.sp_helpdiagrams
	(
		@diagramname sysname = NULL,
		@owner_id int = NULL
	)
	WITH EXECUTE AS N'dbo'
	AS
	BEGIN
		DECLARE @user sysname
		DECLARE @dboLogin bit
		EXECUTE AS CALLER;
			SET @user = USER_NAME();
			SET @dboLogin = CONVERT(bit,IS_MEMBER('db_owner'));
		REVERT;
		SELECT
			[Database] = DB_NAME(),
			[Name] = name,
			[ID] = diagram_id,
			[Owner] = USER_NAME(principal_id),
			[OwnerID] = principal_id
		FROM
			sysdiagrams
		WHERE
			(@dboLogin = 1 OR USER_NAME(principal_id) = @user) AND
			(@diagramname IS NULL OR name = @diagramname) AND
			(@owner_id IS NULL OR principal_id = @owner_id)
		ORDER BY
			4, 5, 1
	END
	;

DROP PROCEDURE IF EXISTS dbo.sp_renamediagram;

	CREATE PROCEDURE dbo.sp_renamediagram
	(
		@diagramname 		sysname,
		@owner_id		int	= null,
		@new_diagramname	sysname
	
	)
	WITH EXECUTE AS 'dbo'
	AS
	BEGIN
		set nocount on
		declare @theId 			int
		declare @IsDbo 			int
		
		declare @UIDFound 		int
		declare @DiagId			int
		declare @DiagIdTarg		int
		declare @u_name			sysname
		if((@diagramname is null) or (@new_diagramname is null))
		begin
			RAISERROR ('Invalid value', 16, 1);
			return -1
		end
	
		EXECUTE AS CALLER;
		select @theId = DATABASE_PRINCIPAL_ID();
		select @IsDbo = IS_MEMBER(N'db_owner'); 
		if(@owner_id is null)
			select @owner_id = @theId;
		REVERT;
	
		select @u_name = USER_NAME(@owner_id)
	
		select @DiagId = diagram_id, @UIDFound = principal_id from dbo.sysdiagrams where principal_id = @owner_id and name = @diagramname 
		if(@DiagId IS NULL or (@IsDbo = 0 and @UIDFound <> @theId))
		begin
			RAISERROR ('Diagram does not exist or you do not have permission.', 16, 1)
			return -3
		end
	
		-- if((@u_name is not null) and (@new_diagramname = @diagramname))	-- nothing will change
		--	return 0;
	
		if(@u_name is null)
			select @DiagIdTarg = diagram_id from dbo.sysdiagrams where principal_id = @theId and name = @new_diagramname
		else
			select @DiagIdTarg = diagram_id from dbo.sysdiagrams where principal_id = @owner_id and name = @new_diagramname
	
		if((@DiagIdTarg is not null) and  @DiagId <> @DiagIdTarg)
		begin
			RAISERROR ('The name is already used.', 16, 1);
			return -2
		end		
	
		if(@u_name is null)
			update dbo.sysdiagrams set [name] = @new_diagramname, principal_id = @theId where diagram_id = @DiagId
		else
			update dbo.sysdiagrams set [name] = @new_diagramname where diagram_id = @DiagId
		return 0
	END
	;

DROP PROCEDURE IF EXISTS dbo.sp_upgraddiagrams;

	CREATE PROCEDURE dbo.sp_upgraddiagrams
	AS
	BEGIN
		IF OBJECT_ID(N'dbo.sysdiagrams') IS NOT NULL
			return 0;
	
		CREATE TABLE dbo.sysdiagrams
		(
			name sysname NOT NULL,
			principal_id int NOT NULL,	-- we may change it to varbinary(85)
			diagram_id int PRIMARY KEY IDENTITY,
			version int,
	
			definition varbinary(max)
			CONSTRAINT UK_principal_name UNIQUE
			(
				principal_id,
				name
			)
		);


		/* Add this if we need to have some form of extended properties for diagrams */
		/*
		IF OBJECT_ID(N'dbo.sysdiagram_properties') IS NULL
		BEGIN
			CREATE TABLE dbo.sysdiagram_properties
			(
				diagram_id int,
				name sysname,
				value varbinary(max) NOT NULL
			)
		END
		*/

		IF OBJECT_ID(N'dbo.dtproperties') IS NOT NULL
		begin
			insert into dbo.sysdiagrams
			(
				[name],
				[principal_id],
				[version],
				[definition]
			)
			select	 
				convert(sysname, dgnm.[uvalue]),
				DATABASE_PRINCIPAL_ID(N'dbo'),			-- will change to the sid of sa
				0,							-- zero for old format, dgdef.[version],
				dgdef.[lvalue]
			from dbo.[dtproperties] dgnm
				inner join dbo.[dtproperties] dggd on dggd.[property] = 'DtgSchemaGUID' and dggd.[objectid] = dgnm.[objectid]	
				inner join dbo.[dtproperties] dgdef on dgdef.[property] = 'DtgSchemaDATA' and dgdef.[objectid] = dgnm.[objectid]
				
			where dgnm.[property] = 'DtgSchemaNAME' and dggd.[uvalue] like N'_EA3E6268-D998-11CE-9454-00AA00A3F36E_' 
			return 2;
		end
		return 1;
	END
	;

DROP Function IF EXISTS dbo.fn_diagramobjects;

	CREATE FUNCTION dbo.fn_diagramobjects() 
	RETURNS int
	WITH EXECUTE AS N'dbo'
	AS
	BEGIN
		declare @id_upgraddiagrams		int
		declare @id_sysdiagrams			int
		declare @id_helpdiagrams		int
		declare @id_helpdiagramdefinition	int
		declare @id_creatediagram	int
		declare @id_renamediagram	int
		declare @id_alterdiagram 	int 
		declare @id_dropdiagram		int
		declare @InstalledObjects	int

		select @InstalledObjects = 0

		select 	@id_upgraddiagrams = object_id(N'dbo.sp_upgraddiagrams'),
			@id_sysdiagrams = object_id(N'dbo.sysdiagrams'),
			@id_helpdiagrams = object_id(N'dbo.sp_helpdiagrams'),
			@id_helpdiagramdefinition = object_id(N'dbo.sp_helpdiagramdefinition'),
			@id_creatediagram = object_id(N'dbo.sp_creatediagram'),
			@id_renamediagram = object_id(N'dbo.sp_renamediagram'),
			@id_alterdiagram = object_id(N'dbo.sp_alterdiagram'), 
			@id_dropdiagram = object_id(N'dbo.sp_dropdiagram')

		if @id_upgraddiagrams is not null
			select @InstalledObjects = @InstalledObjects + 1
		if @id_sysdiagrams is not null
			select @InstalledObjects = @InstalledObjects + 2
		if @id_helpdiagrams is not null
			select @InstalledObjects = @InstalledObjects + 4
		if @id_helpdiagramdefinition is not null
			select @InstalledObjects = @InstalledObjects + 8
		if @id_creatediagram is not null
			select @InstalledObjects = @InstalledObjects + 16
		if @id_renamediagram is not null
			select @InstalledObjects = @InstalledObjects + 32
		if @id_alterdiagram  is not null
			select @InstalledObjects = @InstalledObjects + 64
		if @id_dropdiagram is not null
			select @InstalledObjects = @InstalledObjects + 128
		
		return @InstalledObjects 
	END
	;