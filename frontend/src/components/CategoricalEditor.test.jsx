import React, { useEffect, useState } from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import CategoricalEditor from './CategoricalEditor';

const buildDetails = (overrides = {}) => ({
  categories: ['Male', 'Female'],
  categories_preview: [],
  selected_categories: ['Male', 'Female'],
  custom_categories: [],
  value_counts: { Male: 3, Female: 2 },
  category_null_token: null,
  excluded_strategy: 'map_to_special',
  special_token: '__OTHER__',
  ...overrides,
});

const renderEditor = (overrides = {}) => {
  const changeSpy = vi.fn();
  const latestDetails = { current: null };
  const initialDetails = buildDetails(overrides);

  const Wrapper = () => {
    const [details, setDetails] = useState(initialDetails);

    useEffect(() => {
      latestDetails.current = details;
    }, [details]);

    const handleChange = (updated) => {
      setDetails(updated);
      changeSpy(updated);
    };

    return (
      <CategoricalEditor
        columnKey="gender"
        domainDetails={details}
        onChange={handleChange}
      />
    );
  };

  const utils = render(<Wrapper />);
  return { changeSpy, latestDetails, ...utils };
};

describe('CategoricalEditor', () => {
  afterEach(() => {
    cleanup();
  });

  it('adds a new custom category and keeps it selected', () => {
    const { changeSpy } = renderEditor();

    fireEvent.change(screen.getByPlaceholderText('Add new category'), { target: { value: 'VIP' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));

    expect(screen.getByText('VIP')).toBeInTheDocument();
    expect(changeSpy).toHaveBeenCalled();
    const latestCall = changeSpy.mock.calls.at(-1)[0];
    expect(latestCall.custom_categories).toContain('VIP');
    expect(latestCall.selected_categories).toContain('VIP');
  });

  it('prevents duplicate custom categories regardless of case', () => {
    const { changeSpy } = renderEditor();

    fireEvent.change(screen.getByPlaceholderText('Add new category'), { target: { value: 'female' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));

    expect(changeSpy).not.toHaveBeenCalled();
    expect(screen.getByTestId('category-helper-message')).toHaveTextContent('"female" already exists');
  });

  it('clears validation message after a successful add', () => {
    renderEditor();

    fireEvent.change(screen.getByPlaceholderText('Add new category'), { target: { value: ' ' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));
    expect(screen.getByTestId('category-helper-message')).toHaveTextContent('Enter a non-empty');

    fireEvent.change(screen.getByPlaceholderText('Add new category'), { target: { value: 'TeamA' } });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));

    expect(screen.queryByTestId('category-helper-message')).toBeNull();
    expect(screen.getByText('TeamA')).toBeInTheDocument();
  });
});
