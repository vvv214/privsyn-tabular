import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import SynthesisForm from './SynthesisForm';

const buildProps = (overrides = {}) => {
  const formData = {
    dataset_name: 'adult',
    epsilon: 1.0,
    delta: 1e-5,
    n_sample: 10,
    method: 'privsyn',
    update_iterations: 2,
    num_preprocess: 'uniform_kbins',
    rare_threshold: 0.002,
    consist_iterations: 3,
    append: true,
    sep_syn: false,
    initialize_method: 'singleton',
    ...overrides.formData,
  };

  const submitSpy = vi.fn();
  const props = {
    formData,
    handleChange: overrides.handleChange || vi.fn(),
    handleFileChange: overrides.handleFileChange || vi.fn(),
    handleSubmit: overrides.handleSubmit || ((event) => {
      event.preventDefault();
      submitSpy();
    }),
    handleLoadSample: overrides.handleLoadSample || vi.fn(),
    loadingSample: overrides.loadingSample ?? false,
    isSubmitting: overrides.isSubmitting ?? false,
  };

  return { props, submitSpy };
};

describe('SynthesisForm', () => {
  afterEach(() => cleanup());

  it('submits the form when the primary button is clicked', () => {
    const { props, submitSpy } = buildProps();
    render(<SynthesisForm {...props} />);

    const form = screen.getByText(/infer metadata/i).closest('form');
    fireEvent.submit(form);

    expect(submitSpy).toHaveBeenCalledTimes(1);
  });

  it('disables inputs while submitting', () => {
    const { props } = buildProps({ isSubmitting: true });
    render(<SynthesisForm {...props} />);

    expect(screen.getByLabelText(/dataset name/i)).toBeDisabled();
    expect(screen.getByLabelText(/epsilon/i)).toBeDisabled();
    expect(screen.getByLabelText(/delta/i)).toBeDisabled();
    expect(screen.getByLabelText(/number of samples/i)).toBeDisabled();
    expect(screen.getByLabelText(/upload dataset/i)).toBeDisabled();
    expect(screen.getByRole('button', { name: /load sample/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /inferring/i })).toBeDisabled();
  });

  it('requires file upload only when not using the bundled sample', () => {
    const { props } = buildProps({
      formData: { dataset_name: 'adult' },
    });
    const { rerender } = render(<SynthesisForm {...props} />);

    const fileInput = screen.getByLabelText(/upload dataset/i);
    expect(fileInput).not.toBeRequired();

    const updatedProps = buildProps({ formData: { dataset_name: 'adult_backup' } }).props;
    rerender(<SynthesisForm {...updatedProps} />);
    expect(screen.getByLabelText(/upload dataset/i)).toBeRequired();
  });
});
