def test_init_model():
    test_obj = TransformersSetup()
    test_obj.init_model()
    assert test_obj.tokenizer.is_fast
    with pytest.raises(OSError):
        test_obj.init_model(model_name="Testing")
