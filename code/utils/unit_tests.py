from graph_system.cited_output_nodes import is_complaint_within_4_years

if __name__ == "__main__":
    assert is_complaint_within_4_years("01/01/2020", "01/01/2024") == False 
    assert is_complaint_within_4_years("01/01/2020", "01/01/2026") == False
    assert is_complaint_within_4_years("01/01/2020", "01/03/2024") == False
    assert is_complaint_within_4_years("01/01/2020", "02/01/2024") == False
    assert is_complaint_within_4_years("01/01/2020", "02/03/2024") == False
    assert is_complaint_within_4_years("01/01/2020", "01/01/2023") == True
    assert is_complaint_within_4_years("02/01/2020", "01/01/2024") == True
    assert is_complaint_within_4_years("01/03/2020", "01/01/2024") == True
    assert is_complaint_within_4_years("02/03/2020", "01/01/2024") == True

    print("All tests passed!")