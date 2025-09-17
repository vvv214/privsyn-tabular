import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, within, waitFor, cleanup } from '@testing-library/react';
import MetadataConfirmation from './MetadataConfirmation';

const buildProps = (overrides = {}) => ({
  uniqueId: 'test-id',
  inferredDomainData: {
    gender: {
      type: 'categorical',
      categories: ['Male', 'Female'],
      selected_categories: ['Male', 'Female'],
      custom_categories: [],
      special_token: '__OTHER__',
      value_counts: { Male: 1, Female: 1 },
    },
    age: {
      type: 'numerical',
      bounds: { min: 18, max: 65 },
      binning: {
        method: 'uniform',
        bin_count: 5,
        bin_width: '',
        growth_rate: '',
        dp_budget_fraction: 0.05,
      },
      numeric_summary: { min: 18, max: 65 },
    },
  },
  inferredInfoData: {
    num_columns: ['age'],
    cat_columns: ['gender'],
    n_num_features: 1,
    n_cat_features: 1,
    inference_settings: {},
    inference_report: {},
  },
  onConfirm: vi.fn(),
  onCancel: vi.fn(),
  ...overrides,
});

describe('MetadataConfirmation validation', () => {
  let props;

  beforeEach(() => {
    props = buildProps();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders detected categorical values when only counts are provided', () => {
    const sparseProps = buildProps({
      inferredDomainData: {
        gender: {
          type: 'categorical',
          categories: [],
          categories_preview: [],
          selected_categories: [],
          custom_categories: [],
          special_token: '__OTHER__',
          value_counts: { Male: 3, Female: 2 },
        },
      },
    });

    render(<MetadataConfirmation {...sparseProps} />);

    const genderHeading = screen.getByRole('heading', { name: /^gender$/i });
    const genderCard = genderHeading.closest('.card');
    expect(genderCard).toBeTruthy();

    const options = within(genderCard).getAllByRole('option');
    const optionLabels = options.map((opt) => opt.textContent.trim());
    expect(options.length).toBeGreaterThanOrEqual(2);
    expect(optionLabels).toEqual(expect.arrayContaining([
      'Male (count: 3)',
      'Female (count: 2)',
    ]));
  });

  it('blocks submission when categorical column has no selection', async () => {
    render(<MetadataConfirmation {...props} />);

    const genderHeading = screen.getByRole('heading', { name: /^gender$/i });
    const genderCard = genderHeading.closest('.card');
    const clearButton = within(genderCard).getByRole('button', { name: /clear all/i });
    fireEvent.click(clearButton);

    const listbox = within(genderCard).getByRole('listbox');
    await waitFor(() => {
      const options = within(listbox).getAllByRole('option');
      expect(options.every((opt) => opt.getAttribute('aria-selected') !== 'true')).toBe(true);
    });

    fireEvent.click(screen.getByRole('button', { name: /confirm & synthesize/i }));

    expect(props.onConfirm).not.toHaveBeenCalled();
    expect(screen.getByText(/select at least one category/i)).toBeInTheDocument();
  });

  it('blocks submission when numeric bounds are invalid', async () => {
    render(<MetadataConfirmation {...props} />);

    const ageHeading = screen.getByRole('heading', { name: /^age$/i });
    const ageCard = ageHeading.closest('.card');
    const spinboxes = within(ageCard).getAllByRole('spinbutton');
    const minInput = spinboxes[0];
    const maxInput = spinboxes[1];

    fireEvent.change(minInput, { target: { value: '20' } });
    fireEvent.change(maxInput, { target: { value: '10' } });

    fireEvent.click(screen.getByRole('button', { name: /confirm & synthesize/i }));

    expect(props.onConfirm).not.toHaveBeenCalled();
    expect(
      screen.getByText(/maximum must be greater than or equal to minimum/i)
    ).toBeInTheDocument();
  });

  it('allows submission when data is valid', () => {
    render(<MetadataConfirmation {...props} />);

    fireEvent.click(screen.getByRole('button', { name: /confirm & synthesize/i }));

    expect(props.onConfirm).toHaveBeenCalledTimes(1);
  });
});
