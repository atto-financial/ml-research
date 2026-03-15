public	answers	answer_id	uuid	true	gen_random_uuid()	PK		
public	answers	session_id	character varying(12)	true				
public	answers	user_id	uuid	true		FK	users	user_id
public	answers	feature_id	uuid	true		FK	features	feature_id
public	answers	model_id	uuid	true		FK	models	model_id
public	answers	answer	jsonb	false				
public	answers	ml_artifact	jsonb	false				
public	answers	approval_loan_status	approval_loan_status	false				
public	answers	updated_at	timestamp with time zone	true	now()			
public	answers	created_at	timestamp with time zone	true	now()			
public	collection_fee_histories	collection_fee_id	uuid	true	gen_random_uuid()	PK		
public	collection_fee_histories	user_id	uuid	true		FK	users	user_id
public	collection_fee_histories	loan_ref_id	uuid	true		FK	loan_requests	loan_ref_id
public	collection_fee_histories	loan_due_date	timestamp with time zone	true				
public	collection_fee_histories	round	smallint	true				
public	collection_fee_histories	total_loan_amount	numeric(8,2)	true				
public	collection_fee_histories	total_amount	numeric(8,2)	true				
public	collection_fee_histories	created_at	timestamp with time zone	true	now()			
public	connect_platforms	user_id	uuid	true		PK	users	user_id
public	connect_platforms	line_user_id	character varying(64)	true				
public	connect_platforms	line_latest_profile_display	character varying(25)	false				
public	connect_platforms	line_latest_profile_picture	character varying(260)	false				
public	connect_platforms	line_is_blocked	boolean	true	false			
public	connect_platforms	updated_at	timestamp with time zone	true	now()			
public	connect_platforms	created_at	timestamp with time zone	true	now()			
public	consents	consent_id	uuid	true	gen_random_uuid()	PK		
public	consents	user_id	uuid	true		FK	users	user_id
public	consents	version	character varying(50)	false				
public	consents	content	text	true				
public	consents	allowed	boolean	true	false			
public	consents	created_at	timestamp with time zone	true	now()			
public	default_configs	c_key	character varying(100)	true		PK		
public	default_configs	c_type	type_config	true	'string'::type_config			
public	default_configs	c_value	text	true				
public	feature_groups_v4	group_id	character varying(50)	true		PK		
public	feature_groups_v4	feature_id	character varying(50)	true		FK	features_v4	feature_id
public	feature_groups_v4	group_name	character varying(255)	true				
public	feature_groups_v4	group_code	character varying(100)	true				
public	feature_groups_v4	sort_order	integer	false	0			
public	feature_groups_v4	created_at	timestamp with time zone	true	now()			
public	feature_groups_v4	updated_at	timestamp with time zone	true	now()			
public	feature_item_choices_v4	choice_id	character varying(50)	true		PK		
public	feature_item_choices_v4	item_id	character varying(50)	true		FK	feature_items_v4	item_id
public	feature_item_choices_v4	choice_text	text	true				
public	feature_item_choices_v4	choice_version	character varying(50)	false	'v1.0'::character varying			
public	feature_item_choices_v4	sort_order	integer	false	0			
public	feature_item_choices_v4	created_at	timestamp with time zone	true	now()			
public	feature_item_choices_v4	value	integer	false	0			
public	feature_item_choices_v4	mask	character varying(20)	false	'neutral'::character varying			
public	feature_items_v4	item_id	character varying(50)	true		PK		
public	feature_items_v4	group_id	character varying(50)	true		FK	feature_groups_v4	group_id
public	feature_items_v4	item_name	character varying(255)	true				
public	feature_items_v4	item_code	character varying(100)	true				
public	feature_items_v4	sort_order	integer	false	0			
public	feature_items_v4	active_version_code	character varying(50)	false	'v1.0'::character varying			
public	feature_items_v4	question_text	text	true				
public	feature_items_v4	question_type	character varying(50)	false	'multiple_choice'::character varying			
public	feature_items_v4	created_at	timestamp with time zone	true	now()			
public	feature_items_v4	updated_at	timestamp with time zone	true	now()			
public	features	feature_id	uuid	true	gen_random_uuid()	PK		
public	features	feature_slug	character varying(36)	true				
public	features	title	character varying(255)	true				
public	features	description	text	false				
public	features	questions_n_choices	jsonb	false				
public	features	removed	boolean	true	false			
public	features	updated_by	character varying(36)	true				
public	features	created_by	character varying(36)	true				
public	features	created_at	timestamp with time zone	true	now()			
public	features	updated_at	timestamp with time zone	true	now()			
public	features_v4	feature_id	character varying(50)	true		PK		
public	features_v4	title	character varying(255)	true				
public	features_v4	slug	character varying(100)	true				
public	features_v4	description	text	false				
public	features_v4	sort_order	integer	false	0			
public	features_v4	created_at	timestamp with time zone	true	now()			
public	features_v4	updated_at	timestamp with time zone	true	now()			
public	features_v4	version	character varying(50)	false	'v1.0'::character varying			
public	fine_adjustment_histories	fine_adj_id	uuid	true	gen_random_uuid()	PK		
public	fine_adjustment_histories	user_id	uuid	true		FK	users	user_id
public	fine_adjustment_histories	loan_ref_id	uuid	true		FK	loan_requests	loan_ref_id
public	fine_adjustment_histories	fee	numeric(8,2)	true				
public	fine_adjustment_histories	total_amount	numeric(8,2)	true				
public	fine_adjustment_histories	created_at	timestamp with time zone	true	now()			
public	fine_adjustment_histories	total_loan_amount	numeric(8,2)	true				
public	interest_adjustment_histories	interest_adj_id	uuid	true	gen_random_uuid()	PK		
public	interest_adjustment_histories	user_id	uuid	true		FK	users	user_id
public	interest_adjustment_histories	loan_ref_id	uuid	true		FK	loan_requests	loan_ref_id
public	interest_adjustment_histories	principle	numeric(8,2)	true				
public	interest_adjustment_histories	interest_rate	numeric(6,5)	true				
public	interest_adjustment_histories	total_amount	numeric(8,2)	true				
public	interest_adjustment_histories	created_at	timestamp with time zone	true	now()			
public	interest_adjustment_histories	total_loan_amount	numeric(8,2)	true				
public	loan_action_histories	action_id	uuid	true	gen_random_uuid()	PK		
public	loan_action_histories	user_id	uuid	true		FK	users	user_id
public	loan_action_histories	loan_ref_id	uuid	true		FK	loan_requests	loan_ref_id
public	loan_action_histories	action	loan_action	true				
public	loan_action_histories	amount	numeric(8,2)	true				
public	loan_action_histories	latest_payoff_score	integer	true				
public	loan_action_histories	latest_payoff_count	integer	true				
public	loan_action_histories	attach	character varying(64)	true				
public	loan_action_histories	accepted	boolean	false				
public	loan_action_histories	note	text	true	''::text			
public	loan_action_histories	updated_by	character varying(36)	false				
public	loan_action_histories	updated_at	timestamp with time zone	true	now()			
public	loan_action_histories	created_by	character varying(36)	false				
public	loan_action_histories	created_at	timestamp with time zone	true	now()			
public	loan_groups	loan_group_id	uuid	true	gen_random_uuid()	PK		
public	loan_groups	code	character varying(255)	true				
public	loan_groups	title	character varying(255)	true				
public	loan_groups	description	text	true				
public	loan_groups	logic_condition	jsonb	true				
public	loan_groups	fee_rate	numeric(3,2)	true				
public	loan_groups	interest_rate	numeric(3,2)	true				
public	loan_groups	initial_loan_amount	numeric(8,2)	true	0			
public	loan_groups	initial_max_payoff_day	integer	true	0			
public	loan_groups	updated_by	character varying(32)	false				
public	loan_groups	updated_at	timestamp with time zone	true	now()			
public	loan_groups	created_at	timestamp with time zone	true	now()			
public	loan_requests	loan_ref_id	uuid	true	gen_random_uuid()	PK		
public	loan_requests	user_id	uuid	true		FK	users	user_id
public	loan_requests	loan_id	character varying(32)	true				
public	loan_requests	request_status	loan_request_status	true	'pending'::loan_request_status			
public	loan_requests	approved_loan_at	timestamp with time zone	false				
public	loan_requests	due_date	timestamp with time zone	false				
public	loan_requests	request_amount	numeric(8,2)	true				
public	loan_requests	approved_amount	numeric(8,2)	false				
public	loan_requests	qr_received_img	character varying(64)	false				
public	loan_requests	note	text	true	''::text			
public	loan_requests	updated_by	character varying(36)	false				
public	loan_requests	updated_at	timestamp with time zone	true	now()			
public	loan_requests	created_at	timestamp with time zone	true	now()			
public	loan_requests	total_loan_amount	numeric(8,2)	false				
public	loan_summary_statuses	user_id	uuid	true		PK	users	user_id
public	loan_summary_statuses	loanable	boolean	false				
public	loan_summary_statuses	fee_rate	numeric(3,2)	true	0			
public	loan_summary_statuses	interest_rate	numeric(3,2)	true	0			
public	loan_summary_statuses	payoff_count	integer	true	0			
public	loan_summary_statuses	principle	numeric(8,2)	true	0			
public	loan_summary_statuses	total_loan_amount	numeric(8,2)	false				
public	loan_summary_statuses	loan_status	loan_status	false	'never'::loan_status			
public	loan_summary_statuses	payoff_score	integer	true	0			
public	loan_summary_statuses	max_loan_amount	numeric(8,2)	true	0			
public	loan_summary_statuses	max_payoff_day	integer	true	0			
public	loan_summary_statuses	cool_down_payoff_count	integer	true	0			
public	loan_summary_statuses	cool_down_payoff_date	timestamp with time zone	true	now()			
public	loan_summary_statuses	updated_at	timestamp with time zone	true	now()			
public	loan_summary_statuses	loan_group_id	uuid	false		FK	loan_groups	loan_group_id
public	metric_daily_snapshots	created_at	date	false				
public	metric_daily_snapshots	mkt_overall_users	numeric	false				
public	metric_daily_snapshots	mkt_repeat_loan_users	numeric	false				
public	metric_daily_snapshots	mkt_block_users	numeric	false				
public	metric_daily_snapshots	ml_healthy_users	numeric	false				
public	metric_daily_snapshots	ml_overdue_users	numeric	false				
public	metric_daily_snapshots	ml_npl_users	numeric	false				
public	metric_daily_snapshots	dc_overdue_amount	numeric	false				
public	metric_daily_snapshots	dc_npl_amount	numeric	false				
public	metric_daily_snapshots	dc_overdue_payback	numeric	false				
public	metric_daily_snapshots	dc_npl_payback	numeric	false				
public	metric_daily_snapshots	fin_expected_gain	numeric	false				
public	metric_daily_snapshots	fin_realized_gain	numeric	false				
public	metric_daily_snapshots	mds_id	smallint	false				
public	metric_finance_track	month_date	date	false				
public	metric_finance_track	revenue_amount	numeric	false				
public	metric_finance_track	invested_amount	numeric	false				
public	metric_finance_track	default_amount	numeric	false				
public	metric_finance_track	operate_cost_amoount	numeric	false				
public	metric_finance_track	terminate_portfollio_amount	numeric	false				
public	metric_finance_track	acc_offset_amount	numeric	false				
public	model_predicts	feature_id	uuid	true		PK	features	feature_id
public	model_predicts	session_id	character varying(12)	true		PK	answers	session_id
public	model_predicts	user_id	uuid	true		PK	users	user_id
public	model_predicts	model_id	uuid	true		PK	models	model_id
public	model_predicts	default_probability	numeric(4,3)	false				
public	model_predicts	model_prediction	smallint	false				
public	model_predicts	adjust_prediction	smallint	false				
public	model_predicts	data_set	data_set	true	'do_nothing'::data_set			
public	model_predicts	nextgen_prediction	smallint	false				
public	model_predicts	nextgen_adjprediction	smallint	false				
public	model_predicts	updated_at	timestamp with time zone	true	now()			
public	model_predicts	created_at	timestamp with time zone	true	now()			
public	models	model_id	uuid	true	gen_random_uuid()	PK		
public	models	model_slug	character varying(36)	true				
public	models	title	character varying(255)	true				
public	models	description	text	false				
public	models	removed	boolean	true	false			
public	models	updated_by	character varying(36)	true				
public	models	updated_at	timestamp with time zone	true	now()			
public	models	created_by	character varying(36)	true				
public	models	created_at	timestamp with time zone	true	now()			
public	questionnaire_assignments	user_type	character varying(50)	true		PK		
public	questionnaire_assignments	questionnaire_id	character varying(50)	true		FK	questionnaires_v4	questionnaire_id
public	questionnaire_assignments	updated_at	timestamp with time zone	false	now()			
public	questionnaire_assignments	feature_id	character varying(50)	false	''::character varying			
public	questionnaire_features_v4	questionnaire_id	character varying(50)	true		PK	questionnaires_v4	questionnaire_id
public	questionnaire_features_v4	feature_id	character varying(50)	true		PK	features_v4	feature_id
public	questionnaire_features_v4	sort_order	integer	false	0			
public	questionnaire_questions	link_id	character varying(255)	true		PK		
public	questionnaire_questions	questionnaire_id	character varying(255)	false		FK	questionnaires	questionnaire_id
public	questionnaire_questions	question_id	character varying(255)	false		FK	questions_v3	question_id
public	questionnaire_questions	display_order	integer	false				
public	questionnaire_questions	is_required	boolean	false	true			
public	questionnaire_questions	created_at	timestamp with time zone	false				
public	questionnaire_response_answers_v4	answer_id	character varying(50)	true		PK		
public	questionnaire_response_answers_v4	response_id	character varying(50)	true		FK	questionnaire_responses_v4	response_id
public	questionnaire_response_answers_v4	feature_id	character varying(50)	true				
public	questionnaire_response_answers_v4	group_id	character varying(50)	true				
public	questionnaire_response_answers_v4	item_id	character varying(50)	true				
public	questionnaire_response_answers_v4	choice_id	character varying(50)	false				
public	questionnaire_response_answers_v4	text_value	text	false				
public	questionnaire_response_answers_v4	sort_order	integer	false	0			
public	questionnaire_response_answers_v4	created_at	timestamp with time zone	true	now()			
public	questionnaire_responses_v4	response_id	character varying(50)	true		PK		
public	questionnaire_responses_v4	questionnaire_id	character varying(50)	true		FK	questionnaires_v4	questionnaire_id
public	questionnaire_responses_v4	user_id	character varying(50)	false				
public	questionnaire_responses_v4	is_completed	boolean	false	true			
public	questionnaire_responses_v4	created_at	timestamp with time zone	true	now()			
public	questionnaire_responses_v4	updated_at	timestamp with time zone	true	now()			
public	questionnaires	questionnaire_id	character varying(255)	true		PK		
public	questionnaires	title	character varying(255)	false				
public	questionnaires	description	text	false				
public	questionnaires	is_active	boolean	false	true			
public	questionnaires	created_by	character varying(255)	false				
public	questionnaires	created_at	timestamp with time zone	false				
public	questionnaires	updated_by	character varying(255)	false				
public	questionnaires	updated_at	timestamp with time zone	false				
public	questionnaires_v4	questionnaire_id	character varying(50)	true		PK		
public	questionnaires_v4	title	character varying(255)	true				
public	questionnaires_v4	description	text	false				
public	questionnaires_v4	is_active	boolean	false	true			
public	questionnaires_v4	created_by	character varying(50)	true				
public	questionnaires_v4	created_at	timestamp with time zone	true	now()			
public	questionnaires_v4	updated_by	character varying(50)	true				
public	questionnaires_v4	updated_at	timestamp with time zone	true	now()			
public	questions	question_id	uuid	true	gen_random_uuid()	PK		
public	questions	group	character varying(36)	true				
public	questions	item	character varying(36)	true				
public	questions	version	character varying(36)	true				
public	questions	question	text	true				
public	questions	choices	jsonb	true				
public	questions	removed	boolean	true	false			
public	questions	created_by	character varying(32)	true				
public	questions	created_at	timestamp with time zone	true	now()			
public	questions_v3	question_id	character varying(255)	true		PK		
public	questions_v3	text	text	false				
public	questions_v3	type	character varying(50)	false				
public	questions_v3	choices	jsonb	false				
public	questions_v3	validation	jsonb	false				
public	questions_v3	created_by	character varying(255)	false				
public	questions_v3	created_at	timestamp with time zone	false				
public	questions_v3	updated_by	character varying(255)	false				
public	questions_v3	updated_at	timestamp with time zone	false				
public	user_academics	user_id	uuid	true		PK	users	user_id
public	user_academics	name	character varying(255)	false				
public	user_academics	academic_years	smallint	false				
public	user_academics	end_date	date	false				
public	user_academics	student_credential_img	character varying(64)	false				
public	user_academics	created_at	timestamp with time zone	true	now()			
public	user_academics	updated_at	timestamp with time zone	true	now()			
public	user_academics	faculty	character varying(550)	false				
public	user_academics	monthly_income	numeric(8,2)	true	0			
public	user_addresses	user_id	uuid	true		PK	users	user_id
public	user_addresses	address	character varying(550)	false				
public	user_addresses	subdistrict	character varying(150)	false				
public	user_addresses	district	character varying(150)	false				
public	user_addresses	province	character varying(250)	false				
public	user_addresses	zipcode	character varying(25)	false				
public	user_addresses	updated_at	timestamp with time zone	true	now()			
public	user_messages	message_id	uuid	true	gen_random_uuid()	PK		
public	user_messages	user_id	uuid	true		FK	users	user_id
public	user_messages	sender_type	character varying(10)	true				
public	user_messages	message_text	text	true				
public	user_messages	line_message_id	character varying(255)	false				
public	user_messages	line_user_id	character varying(255)	false				
public	user_messages	sent_at	timestamp with time zone	true	now()			
public	user_messages	created_at	timestamp with time zone	true	now()			
public	user_ml_training_status_histories	history_id	uuid	true	gen_random_uuid()	PK		
public	user_ml_training_status_histories	user_id	uuid	true		FK	users	user_id
public	user_ml_training_status_histories	previous_status	ml_training_status	false				
public	user_ml_training_status_histories	new_status	ml_training_status_old	true				
public	user_ml_training_status_histories	note	text	true	''::text			
public	user_ml_training_status_histories	changed_by	character varying(36)	true				
public	user_ml_training_status_histories	changed_at	timestamp with time zone	true	now()			
public	user_ml_training_statuses	user_id	uuid	true		PK	users	user_id
public	user_ml_training_statuses	training_status	ml_training_status_old	true				
public	user_ml_training_statuses	updated_by	character varying(36)	false				
public	user_ml_training_statuses	updated_at	timestamp with time zone	true	now()			
public	user_ml_training_statuses	created_by	character varying(36)	false				
public	user_ml_training_statuses	created_at	timestamp with time zone	true	now()			
public	user_occupations	user_id	uuid	true		PK	users	user_id
public	user_occupations	name	character varying(250)	false				
public	user_occupations	monthly_income	numeric(8,2)	true	0			
public	user_occupations	updated_at	timestamp with time zone	true	now()			
public	user_occupations	company_name	character varying(255)	false				
public	user_occupations	company_address	character varying(550)	false				
public	user_occupations	company_sub_district	character varying(150)	false				
public	user_occupations	company_district	character varying(150)	false				
public	user_occupations	company_province	character varying(250)	false				
public	user_occupations	company_zipcode	character varying(25)	false				
public	user_occupations	credential_img	character varying(64)	false				
public	user_onboardings	onboarding_id	uuid	true	gen_random_uuid()	PK		
public	user_onboardings	answer_id	uuid	true		FK	answers	answer_id
public	user_onboardings	user_id	uuid	true		FK	users	user_id
public	user_onboardings	loan_group_id	uuid	false		FK	loan_groups	loan_group_id
public	user_onboardings	status	user_onboading_status	true	'pending'::user_onboading_status			
public	user_onboardings	status_by	character varying(32)	false				
public	user_onboardings	note	text	true	''::text			
public	user_onboardings	updated_at	timestamp with time zone	true	now()			
public	user_onboardings	created_at	timestamp with time zone	true	now()			
public	users	user_id	uuid	true	gen_random_uuid()	PK		
public	users	cid	character varying(64)	false				
public	users	cid_hash	character varying(64)	false				
public	users	birth_date	date	false				
public	users	id_card_img	character varying(64)	false				
public	users	gender	gender	false	'other'::gender			
public	users	user_status	user_status	false	'normal'::user_status			
public	users	updated_at	timestamp with time zone	true	now()			
public	users	created_at	timestamp with time zone	true	now()			
public	users	data_validation_status	user_validation_status	true	'unverified'::user_validation_status			