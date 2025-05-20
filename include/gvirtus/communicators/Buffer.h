/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   Buffer.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 18 13:16:46 2009
 *
 * @brief
 *
 *
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <typeinfo>

#include <gvirtus/common/gvirtus-type.h>

#include "Communicator.h"

#define BLOCK_SIZE 4096

namespace gvirtus::communicators {
/**
 * Buffer is a general purpose for marshalling and unmarshalling data. It's used
 * for exchanging data beetwen Frontend and Backend. It has the functionality to
 * be created starting from an input stream and to be sent over an output
 * stream.
 */
class Buffer {
 public:
  Buffer(size_t initial_size = 0, size_t block_size = BLOCK_SIZE);
  Buffer(const Buffer &orig);
  Buffer(std::istream &in);
  Buffer(char *buffer, size_t buffer_size, size_t block_size = BLOCK_SIZE);
  virtual ~Buffer();

  template <class T>
  void Add(const T& item) {
      if constexpr (std::is_void_v<T>) {
          // You typically can't add a single 'void' value, so either static_assert or no-op
          static_assert(!std::is_void_v<T>, "Buffer::Add<T>: Adding single 'void' item not supported");
      } else {
          size_t itemSize = sizeof(T);
          if ((mLength + itemSize) >= mSize) {
              mSize = ((mLength + itemSize) / mBlockSize + 1) * mBlockSize;
              if ((mpBuffer = (char*)realloc(mpBuffer, mSize)) == NULL)
                  throw "Buffer::Add(item): Can't reallocate memory.";
          }
          memmove(mpBuffer + mLength, &item, itemSize);
          mLength += itemSize;
          mBackOffset = mLength;
      }
  }

  template <class T>
  void Add(const T* item, size_t n = 1) {
      if constexpr (std::is_void_v<T>) {
          // For void*, forward to your existing Add(const void*, size_t)
          if (item == nullptr) {
              Add((size_t)0);
              return;
          }
          size_t size = n; // n here is number of bytes for void*
          Add(static_cast<const void*>(item), size);
      } else {
          static_assert(!std::is_void_v<T>, "Buffer::Add<T*>: void type not supported");
          if (item == nullptr) {
              Add((size_t)0);
              return;
          }
          size_t size = sizeof(T) * n;
          Add(size);
          if ((mLength + size) >= mSize) {
              mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
              if ((mpBuffer = (char*)realloc(mpBuffer, mSize)) == NULL)
                  throw "Buffer::Add(item, n): Can't reallocate memory.";
          }
          memmove(mpBuffer + mLength, item, size);
          mLength += size;
          mBackOffset = mLength;
      }
  }

  // The raw data add method remains the same:
  void Add(const void* data, size_t size) {
      if ((mLength + size) >= mSize) {
          mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
          if ((mpBuffer = (char*)realloc(mpBuffer, mSize)) == NULL)
              throw "Buffer::Add(data, size): Can't reallocate memory.";
      }
      memmove(mpBuffer + mLength, data, size);
      mLength += size;
      mBackOffset = mLength;
  }

  template <class T>
  void AddConst(const T *item, size_t n = 1) {
    if (item == NULL) {
      Add((size_t)0);
      return;
    }
    size_t size = sizeof(T) * n;
    Add(size);
    if ((mLength + size) >= mSize) {
      mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
      if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
        throw "Buffer::AddConst(item, n): Can't reallocate memory.";
    }
    memmove(mpBuffer + mLength, (char *)item, size);
    mLength += size;
    mBackOffset = mLength;
  }

  // Overload if caller passes value instead of pointer
  template <class T>
  void AddConst(const T &item) {
      AddConst(&item, 1);
  }

  void AddString(const char *s) {
    size_t size = strlen(s) + 1;
    Add(size);
    Add(s, size);
  }

  template <class T>
  void AddMarshal(T item) {
    Add((gvirtus::common::pointer_t)item);
  }

  template <class T>
  void Read(Communicator *c) {
    auto required_size = mLength + sizeof(T);
    if (required_size >= mSize) {
      mSize = (required_size / mBlockSize + 1) * mBlockSize;
      if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
        throw "Buffer::Read(*c) Can't reallocate memory.";
    }
    c->Read(mpBuffer + mLength, sizeof(T));
    mLength += sizeof(T);
    mBackOffset = mLength;
  }

  template <class T>
  void Read(Communicator *c, size_t n = 1) {
    auto required_size = mLength + sizeof(T) * n;
    if (required_size >= mSize) {
      mSize = (required_size / mBlockSize + 1) * mBlockSize;
      if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
        throw "Buffer::Read(*c, n): Can't reallocate memory.";
    }
    c->Read(mpBuffer + mLength, sizeof(T) * n);
    mLength += sizeof(T) * n;
    mBackOffset = mLength;
  }

  template <class T>
  T Get() {
    if (mOffset + sizeof(T) > mLength)
      throw "Buffer::Get(): Can't read any " + std::string(typeid(T).name()) + ".";
    T result = *((T *)(mpBuffer + mOffset));
    mOffset += sizeof(T);
    return result;
  }

  template <class T>
  T BackGet() {
    if (mBackOffset - sizeof(T) > mLength)
      throw "Buffer::BackGet(): Can't read  " + std::string(typeid(T).name()) + ".";
    T result = *((T *)(mpBuffer + mBackOffset - sizeof(T)));
    mBackOffset -= sizeof(T);
    return result;
  }

  template <class T>
  T *Get(size_t n) {
    if (Get<size_t>() == 0) return NULL;
    if (mOffset + sizeof(T) * n > mLength)
      throw "Buffer::Get(n): Can't read  " + std::string(typeid(T).name()) + ".";
    T *result = new T[n];
    memmove((char *)result, mpBuffer + mOffset, sizeof(T) * n);
    mOffset += sizeof(T) * n;
    return result;
  }

  template <class T>
  T *Delegate(size_t n = 1) {
    size_t size = sizeof(T) * n;
    Add(size);
    if ((mLength + size) >= mSize) {
      mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
      if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
        throw "Buffer::Delegate(n): Can't reallocate memory.";
    }
    T *dst = (T *)(mpBuffer + mLength);
    mLength += size;
    mBackOffset = mLength;
    return dst;
  }

  template <class T>
  T* Assign(size_t n = 1) {
      if constexpr (std::is_void_v<T>) {
          if (Get<size_t>() == 0) return nullptr;

          if (mOffset + n > mLength) {
              throw std::string("Buffer::Assign(n): Can't read void.");
          }
          void* result = static_cast<void*>(mpBuffer + mOffset);
          mOffset += n;
          return static_cast<T*>(result);  // T is void, so void*
      } else {
          if (Get<size_t>() == 0) return nullptr;

          if (mOffset + sizeof(T) * n > mLength) {
              throw std::string("Buffer::Assign(n): Can't read ") + typeid(T).name() + ".";
          }
          T* result = reinterpret_cast<T*>(mpBuffer + mOffset);
          mOffset += sizeof(T) * n;
          return result;
      }
  }

  template <class T>
  T* AssignAll() {
      size_t size = Get<size_t>();
      if (size == 0) return nullptr;

      if constexpr (std::is_void_v<T>) {
          if (mOffset + size > mLength) {
              throw std::string("Buffer::AssignAll(): Can't read void.");
          }
          void* result = static_cast<void*>(mpBuffer + mOffset);
          mOffset += size;
          return static_cast<T*>(result);
      } else {
          size_t n = size / sizeof(T);
          if (mOffset + sizeof(T) * n > mLength) {
              throw std::string("Buffer::AssignAll(): Can't read ") + typeid(T).name() + ".";
          }
          T* result = reinterpret_cast<T*>(mpBuffer + mOffset);
          mOffset += sizeof(T) * n;
          return result;
      }
  }

  char *AssignString() {
    size_t size = Get<size_t>();
    return Assign<char>(size);
  }

  template <class T>
  T *BackAssign(size_t n = 1) {
    if (mBackOffset - sizeof(T) * n > mLength)
      throw "Buffer::BackAssign(n): Can't read  " + std::string(typeid(T).name()) + ".";
    T *result = (T *)(mpBuffer + mBackOffset - sizeof(T) * n);
    mBackOffset -= sizeof(T) * n + sizeof(size_t);
    return result;
  }

  template <class T>
  T GetFromMarshal() {
    return (T)Get<gvirtus::common::pointer_t>();
  }

  inline bool Empty() { return mOffset == mLength; }

  void Reset();
  void Reset(Communicator *c);
  const char *const GetBuffer() const;
  size_t GetBufferSize() const;
  void Dump(Communicator *c) const;

 private:
  size_t mBlockSize;
  size_t mSize;
  size_t mLength;
  size_t mOffset;
  size_t mBackOffset;
  char *mpBuffer;
  bool mOwnBuffer;
};
}  // namespace gvirtus::communicators
