import great_expectations as gx
import pandas as pd

print(gx.__version__)  

def run_phishing_ge_validation(bank_csv_path):
    """
    Run Great Expectations validation on the bank dataset.
    """

    df = pd.read_csv(bank_csv_path, sep=';')  
    context = gx.get_context()

    context.add_datasource(
        name="bank_pandas_datasource",
        class_name="Datasource",
        execution_engine={
            "class_name": "PandasExecutionEngine"
        },
        data_connectors={
            "default_runtime_data_connector_name": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier_name"]
            }
        }
    )

    print(f"Context type: {type(context).__name__}")

    suite_name = "bank_expectation_suite"
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    batch_request = gx.core.batch.RuntimeBatchRequest(
        datasource_name="bank_pandas_datasource",
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="bank_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default_identifier"}
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    for column in ['age', 'job', 'marital', 'education', 'default', 'housing', 
                   'loan', 'contact', 'month', 'day_of_week', 'duration', 
                   'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 
                   'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']:
        validator.expect_column_to_exist(column)

    for column in ['age', 'duration', 'campaign', 'y']:
        validator.expect_column_values_to_not_be_null(column)

    validator.expect_column_values_to_be_between(
        column="age",
        min_value=18,
        max_value=100,
        mostly=0.99
    )

    validator.expect_column_distinct_values_to_be_in_set(
        column="job",
        value_set=['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                  'retired', 'self-employed', 'services', 'student', 'technician', 
                  'unemployed', 'unknown']
    )

    validator.expect_column_values_to_be_in_set(
        column="marital",
        value_set=['married', 'single', 'divorced', 'unknown']
    )

    validator.expect_column_values_to_be_in_set(
        column="education",
        value_set=['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                  'illiterate', 'professional.course', 'university.degree', 'unknown']
    )

    validator.expect_column_values_to_be_between(
        column="duration",
        min_value=0,
        mostly=1.0
    )

    validator.expect_column_values_to_be_between(
        column="campaign",
        min_value=1,
        mostly=1.0
    )

    validator.expect_column_values_to_be_in_set(
        column="pdays",
        value_set=list(range(0, 1000)) + [999]
    )

    validator.expect_column_values_to_be_between(
        column="previous",
        min_value=0,
        mostly=1.0
    )

    validator.expect_column_values_to_be_in_set(
        column="poutcome",
        value_set=['success', 'failure', 'nonexistent', 'other']
    )

    validator.expect_column_values_to_be_in_set(
        column="y",
        value_set=['yes', 'no']
    )

    validator.save_expectation_suite(discard_failed_expectations=False)
    validation_result = validator.validate()

    print("Validation success:", validation_result.success)
    print("Validation statistics:", validation_result.statistics)

    print("\n Detailed Expectation Results:\n")
    for idx, result in enumerate(validation_result.results, 1):
        expectation_type = result.expectation_config.expectation_type
        kwargs = result.expectation_config.kwargs
        success = result.success

        print(f"🔹 Expectation {idx}: {expectation_type}")
        print(f"   ➤ Kwargs: {kwargs}")
        print(f"    Success: {success}")
        
        if not success:
            unexpected = result.result.get("unexpected_list", None)
            if unexpected:
                print(f"   Unexpected values: {unexpected}")
            elif "unexpected_percent" in result.result:
                print(f"  Unexpected %: {result.result['unexpected_percent']:.2f}%")
            elif "element_count" in result.result:
                print(f"  Details: {result.result}")
        print("-" * 80)

    # Convert the validation result to a JSON-serializable dictionary
    return validation_result.to_json_dict()